"""Run YOLO segmentation on a webcam stream or other OpenCV source."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from ultralytics import YOLO

# 스크립트가 위치한 경로를 기준으로 기본 모델 경로를 고정해 다른 PC에서도 동일한 구조를 유지한다.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "runs" / "detect" / "train_fixed_aug" / "weights" / "best.pt"
DEFAULT_SOURCE: Union[int, Path] = 0

# 형태학 연산으로 세그멘테이션 마스크를 다듬어주는 함수
def refine_segmentation_mask(binary_mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Return a shrunken mask that better hugs the underlying object."""

    mask_255 = (binary_mask > 0).astype(np.uint8) * 255

    cleaned = cv2.morphologyEx(mask_255, cv2.MORPH_OPEN, kernel)
    refined = cv2.erode(cleaned, kernel, iterations=1)
    return (refined > 0).astype(np.uint8)

# 경로에 따라 YOLO 모델을 로드하는 함수
def load_model(model_path: Path) -> YOLO:
    if model_path.suffix.lower() == ".onnx":
        return YOLO(str(model_path))
    return YOLO(str(model_path))

# 비디오 소스를 열고 실패 시 명확한 안내 메시지를 제공하는 함수
def _open_capture(source: Union[int, Path]) -> cv2.VideoCapture:
    """Return a video capture object for the given source, raising on failure."""

    if isinstance(source, Path):
        capture = cv2.VideoCapture(str(source))
    else:
        capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            "지정한 비디오 소스를 열 수 없습니다. 웹캠이 없는 장비라면 --source 옵션으로 영상 파일 경로를 지정하세요."
        )

    return capture


# 웹캠 영상에 대해 실시간 감지를 수행하는 함수
def webcam_detection(
    model_path: Path = DEFAULT_MODEL, source: Union[int, Path] = DEFAULT_SOURCE
) -> None:
    model = load_model(model_path)
    cap = _open_capture(source)

    alpha = 0.4

    class_colors = {
        "cast1": (0, 0, 255),
        "cast2": (0, 255, 0),
        "cast3": (255, 0, 0),
    }

    refine_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        frame_result = results[0]
        masks = frame_result.masks

        annotated_frame = frame.copy()
        if masks is not None:
            overlay = annotated_frame.copy()
            labels_to_draw = []

            for seg, cls in zip(masks.data, frame_result.boxes.cls):
                mask = seg.cpu().numpy()
                binary_mask = (mask > 0.5).astype(np.uint8)
                refined_mask = refine_segmentation_mask(binary_mask, refine_kernel)

                if not np.any(refined_mask):
                    continue

                if int(refined_mask.sum()) < 30:
                    continue

                label = frame_result.names[int(cls)]

                color = class_colors.get(label, (0, 255, 255))
                overlay[refined_mask == 1] = color

                ys, xs = np.nonzero(refined_mask)
                if len(xs) and len(ys):
                    labels_to_draw.append((label, (int(xs.mean()), int(ys.mean()))))

            annotated_frame = cv2.addWeighted(annotated_frame, 1 - alpha, overlay, alpha, 0)

            for label, position in labels_to_draw:
                cv2.putText(
                    annotated_frame,
                    label,
                    position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        cv2.imshow("Webcam Detection (Seg Only, Transparent)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# 커맨드라인 인자를 파싱하는 함수
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run webcam inference using a YOLO model.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to a YOLO model (.pt or .onnx).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(DEFAULT_SOURCE),
        help="Video source: webcam index (e.g. 0) or path to a video file.",
    )
    return parser.parse_args()

# 스크립트 실행 흐름을 제어하는 메인 함수
def main() -> None:
    args = parse_args()
    if args.source.isdigit():
        source_arg: Union[int, Path] = int(args.source)
    else:
        source_arg = Path(args.source)

    try:
        webcam_detection(model_path=args.model_path, source=source_arg)
    except RuntimeError as exc:
        # 웹캠이 없거나 접근 권한이 없을 때 사용자에게 안내 메시지를 보여주고 종료한다.
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
