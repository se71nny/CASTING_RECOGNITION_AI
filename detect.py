"""Run batch inference on the test dataset using a trained YOLO model."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Optional

from ultralytics import YOLO

ENV_BASE_DIR = "CASTING_AI_BASE_DIR"
DEFAULT_MODEL_PATH = Path("runs/detect/train_fixed_aug/weights/best.pt")
DEFAULT_PROJECT_SUBDIR = Path("runs/detect")
DEFAULT_RUN_NAME = "predict_fixed_aug"
DEFAULT_IMAGE_DIR = Path("datasets/test/images")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def resolve_base_dir(base_dir: Optional[Path]) -> Path:
    if base_dir is not None:
        return Path(base_dir).expanduser()

    env_base_dir = os.getenv(ENV_BASE_DIR)
    if env_base_dir:
        return Path(env_base_dir).expanduser()

    return Path(__file__).resolve().parent


def resolve_path(path: Path | str, base_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate


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
    return YOLO(str(model_path))


# YOLO 모델로 배치 추론을 수행하고 결과를 저장하는 함수
def run_inference(
    *,
    base_dir: Optional[Path] = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    image_dir: Path = DEFAULT_IMAGE_DIR,
    project_dir: Path = DEFAULT_PROJECT_SUBDIR,
    run_name: str = DEFAULT_RUN_NAME,
    show: bool = True,
    overwrite: bool = True,
) -> None:
    resolved_base_dir = resolve_base_dir(base_dir)
    project_dir = resolve_path(project_dir, resolved_base_dir)
    image_dir = resolve_path(image_dir, resolved_base_dir)
    model_path = resolve_path(model_path, resolved_base_dir)
    save_path = project_dir / run_name

    project_dir.mkdir(parents=True, exist_ok=True)
    if overwrite and save_path.exists():
        shutil.rmtree(save_path)
    elif not overwrite and save_path.exists():
        raise FileExistsError(
            f"Prediction output directory already exists: {save_path}. "
            "Use --run-name or enable overwriting to continue."
        )

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = load_model(model_path)
    image_paths = list_images(image_dir)
    results = model(
        [str(image) for image in image_paths],
        project=str(project_dir),
        name=run_name,
        save=True,
    )

    if show:
        for result in results:
            result.show()


# 커맨드라인 인자를 파싱하는 함수
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on a directory of images.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help=(
            "Base directory used to resolve relative paths. "
            "Defaults to $CASTING_AI_BASE_DIR or the script directory."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Relative path to a YOLO model (.pt or .onnx).",
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
        default=DEFAULT_PROJECT_SUBDIR,
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
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete an existing prediction directory.",
    )
    return parser.parse_args()


# 스크립트 실행 흐름을 담당하는 메인 함수
def main() -> None:
    args = parse_args()
    run_inference(
        base_dir=args.base_dir,
        model_path=args.model_path,
        image_dir=args.image_dir,
        project_dir=args.project_dir,
        run_name=args.run_name,
        show=not args.no_show,
        overwrite=not args.keep_existing,
    )


if __name__ == "__main__":
    main()
