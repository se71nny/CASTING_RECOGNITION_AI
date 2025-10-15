"""Run batch inference on the test dataset using a trained YOLO model."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List

from ultralytics import YOLO

DEFAULT_MODEL = Path("runs/detect/train_fixed_aug/weights/best.pt")
DEFAULT_PROJECT_DIR = Path("runs/detect")
DEFAULT_RUN_NAME = "predict_fixed_aug"
DEFAULT_IMAGE_DIR = Path("datasets/test/images")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# 지정한 폴더에서 추론에 사용할 이미지 목록을 만드는 함수
def list_images(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    images = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise RuntimeError(f"No images were found in: {image_dir}")
    return images

# 경로에 맞춰 YOLO 모델을 로드하는 함수
def load_model(model_path: Path) -> YOLO:
    if model_path.suffix.lower() == ".onnx":
        return YOLO(str(model_path))
    return YOLO(str(model_path))

# YOLO 모델로 배치 추론을 수행하고 결과를 저장하는 함수
def run_inference(
    model_path: Path = DEFAULT_MODEL,
    image_dir: Path = DEFAULT_IMAGE_DIR,
    project_dir: Path = DEFAULT_PROJECT_DIR,
    run_name: str = DEFAULT_RUN_NAME,
    show: bool = True,
) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    save_path = project_dir / run_name

    if save_path.exists():
        shutil.rmtree(save_path)

    model = load_model(model_path)
    image_paths = list_images(image_dir)
    results = model([str(image) for image in image_paths], project=str(project_dir), name=run_name, save=True)

    if show:
        for result in results:
            result.show()

# 커맨드라인 인자를 파싱하는 함수
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on a directory of images.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to a YOLO model (.pt or .onnx).",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory that contains images to run inference on.",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=DEFAULT_PROJECT_DIR,
        help="Directory where prediction results will be stored.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=DEFAULT_RUN_NAME,
        help="Name of the prediction run directory.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable interactive result windows.",
    )
    return parser.parse_args()

# 스크립트 실행 흐름을 담당하는 메인 함수
def main() -> None:
    args = parse_args()
    run_inference(
        model_path=args.model_path,
        image_dir=args.image_dir,
        project_dir=args.project_dir,
        run_name=args.run_name,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
