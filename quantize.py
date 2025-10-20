"""Utilities for exporting a trained YOLO model to ONNX and INT8."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np
from ultralytics import YOLO


if TYPE_CHECKING:  # pragma: no cover - used for type checkers only
    from onnxruntime import InferenceSession


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_WEIGHTS = Path("runs/detect/train_fixed_aug/weights/best.pt")
DEFAULT_CALIBRATION_DIR = Path("datasets/test/images")
DEFAULT_IMAGE_SIZE = 640
DEFAULT_ONNX_EXPORT_PATH = DEFAULT_WEIGHTS.with_suffix(".onnx")


class YOLOCalibrationDataReader:
    """Feeds preprocessed images to onnxruntime.quantization.quantize_static."""

    # 보정 이미지 리스트와 입력 정보를 저장하는 생성자
    def __init__(
        self,
        image_paths: Iterable[Path],
        input_name: str,
        image_size: Tuple[int, int],
    ) -> None:
        self.image_paths: List[Path] = list(image_paths)
        self.input_name = input_name
        self.image_size = image_size
        self._index = 0

    # 다음 보정 이미지를 로드해 ORT가 소비할 배치를 반환하는 함수
    def get_next(self) -> Optional[dict]:
        if self._index >= len(self.image_paths):
            return None

        image_path = self.image_paths[self._index]
        self._index += 1

        array = self._load_image(image_path)
        return {self.input_name: array}

    # 데이터 리더의 인덱스를 초기화하는 함수
    def rewind(self) -> None:
        self._index = 0

    # 이미지를 읽고 모델 입력 형태로 전처리하는 함수
    def _load_image(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Unable to read calibration image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target_width, target_height = self.image_size
        resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        tensor = resized.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))  # HWC -> CHW
        tensor = np.expand_dims(tensor, axis=0)  # NCHW
        return tensor

# 보정용 이미지 경로를 수집하고 검증하는 함수
def collect_image_paths(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Calibration image directory not found: {image_dir}")

    image_paths = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_paths:
        raise RuntimeError(f"No calibration images were found in: {image_dir}")

    return image_paths

# 학습된 가중치를 ONNX 형식으로 내보내는 함수
def export_to_onnx(weights_path: Path, imgsz: int) -> Path:
    model = YOLO(str(weights_path))
    exported_path = Path(model.export(format="onnx", imgsz=imgsz))
    return exported_path

# ONNX 입력 텐서의 공간 크기를 계산하는 함수
def resolve_input_size(session: "InferenceSession", default_size: int) -> Tuple[int, int]:
    input_meta = session.get_inputs()[0]
    shape = input_meta.shape

    # 동적 차원을 기본값으로 대체하는 내부 함수
    def _resolve(value):
        return value if isinstance(value, int) and value > 0 else default_size

    if len(shape) >= 4:
        height = _resolve(shape[2])
        width = _resolve(shape[3])
    else:
        height = width = default_size

    return int(width), int(height)

# ONNX 모델을 정적 양자화하여 INT8 모델로 저장하는 함수
def _import_onnxruntime():
    try:
        from onnxruntime import InferenceSession as _InferenceSession
        from onnxruntime.quantization import (
            CalibrationDataReader as _CalibrationDataReader,
            QuantFormat as _QuantFormat,
            QuantType as _QuantType,
            quantize_static as _quantize_static,
        )
    except ImportError as exc:  # pragma: no cover - error path
        raise ImportError(
            "onnxruntime with quantization tools is required to run quantize.py. "
            "Install the dependency with 'pip install onnxruntime onnxruntime-tools'."
        ) from exc

    return (
        _InferenceSession,
        _CalibrationDataReader,
        _QuantFormat,
        _QuantType,
        _quantize_static,
    )


def quantize_to_int8(onnx_path: Path, calibration_dir: Path, output_path: Path, imgsz: int) -> Path:
    InferenceSession, _, QuantFormat, QuantType, quantize_static = _import_onnxruntime()
    session = InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    width, height = resolve_input_size(session, imgsz)
    input_name = session.get_inputs()[0].name

    image_paths = collect_image_paths(calibration_dir)
    data_reader = YOLOCalibrationDataReader(image_paths, input_name, (width, height))

    quantize_static(
        str(onnx_path),
        str(output_path),
        data_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    return output_path

# 커맨드라인 인자를 파싱하는 함수
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a YOLO model to ONNX and INT8.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to the trained YOLO weights (.pt).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Image size used during export.",
    )
    parser.add_argument(
        "--calib-images",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
        help="Directory that contains calibration images.",
    )
    parser.add_argument(
        "--onnx-output",
        type=Path,
        default=None,
        help="Optional path to save the exported ONNX model.",
    )
    parser.add_argument(
        "--int8-output",
        type=Path,
        default=None,
        help="Optional path to save the INT8 quantized ONNX model.",
    )
    return parser.parse_args()

# 전체 내보내기와 양자화 과정을 실행하는 메인 함수
def main() -> None:
    args = parse_args()
    weights_path = args.weights
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    if args.onnx_output:
        target_onnx_path = Path(args.onnx_output)
    elif weights_path == DEFAULT_WEIGHTS:
        target_onnx_path = DEFAULT_ONNX_EXPORT_PATH
    else:
        target_onnx_path = weights_path.with_suffix(".onnx")

    target_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    if target_onnx_path.exists():
        target_onnx_path.unlink()

    onnx_path = export_to_onnx(weights_path, args.imgsz)
    if onnx_path != target_onnx_path:
        if target_onnx_path.exists():
            target_onnx_path.unlink()
        onnx_path.replace(target_onnx_path)
        onnx_path = target_onnx_path

    onnx_path = target_onnx_path

    if args.int8_output:
        int8_path = args.int8_output
    else:
        int8_path = onnx_path.with_name(f"{onnx_path.stem}-int8.onnx")

    int8_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_to_int8(onnx_path, args.calib_images, int8_path, args.imgsz)
    print(f"Exported ONNX model: {onnx_path}")
    print(f"Quantized INT8 model: {int8_path}")


if __name__ == "__main__":
    main()
