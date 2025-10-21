"""Utilities for training the YOLO model used in the project."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

ENV_BASE_DIR = "CASTING_AI_BASE_DIR"
DEFAULT_RUN_NAME = "train_fixed_aug"
DEFAULT_PROJECT_SUBDIR = Path("runs/detect")
DEFAULT_DATA_CONFIG = Path("datasets/data.yaml")
DEFAULT_MODEL_WEIGHTS = "yolov8s-seg.pt"


def resolve_base_dir(base_dir: Optional[Path]) -> Path:
    """Return the directory that relative paths should be resolved against."""
    if base_dir is not None:
        return Path(base_dir).expanduser()

    env_base_dir = os.getenv(ENV_BASE_DIR)
    if env_base_dir:
        return Path(env_base_dir).expanduser()

    script_dir = Path(__file__).resolve().parent

    for candidate in (script_dir, *script_dir.parents):
        dataset_path = candidate / DEFAULT_DATA_CONFIG
        root_level_yaml = candidate / DEFAULT_DATA_CONFIG.name
        if dataset_path.exists() or root_level_yaml.exists():
            return candidate

    return script_dir


def resolve_path(path: Path | str, base_dir: Path) -> Path:
    """Resolve *path* relative to *base_dir* if it is not already absolute."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate


def remove_previous_run(save_path: Path, project_dir: Path) -> None:
    """Remove a previous training run directory if it lives under *project_dir*."""
    if not save_path.exists():
        return

    try:
        save_path.resolve().relative_to(project_dir.resolve())
    except ValueError as exc:  # pragma: no cover - guard clause
        raise RuntimeError(
            f"Refusing to delete path outside {project_dir}: {save_path}"
        ) from exc

    shutil.rmtree(save_path)


# 학습 파이프라인을 구성하고 YOLO 모델을 훈련하는 함수

def train_model(
    *,
    base_dir: Optional[Path] = None,
    data_config: Path = DEFAULT_DATA_CONFIG,
    project_dir: Path = DEFAULT_PROJECT_SUBDIR,
    run_name: str = DEFAULT_RUN_NAME,
    model_weights: str | Path = DEFAULT_MODEL_WEIGHTS,
    epochs: int = 25,
    imgsz: int = 640,
    batch: int = 12,
    auto_augment: Optional[str] = None,
    overwrite: bool = True,
) -> None:
    resolved_base_dir = resolve_base_dir(base_dir)
    project_dir = resolve_path(project_dir, resolved_base_dir)
    save_path = project_dir / run_name

    project_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        remove_previous_run(save_path, project_dir)
    elif save_path.exists():
        raise FileExistsError(
            f"Training output directory already exists: {save_path}. "
            "Use --run-name or enable overwriting to continue."
        )

    data_path = resolve_path(data_config, resolved_base_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    if isinstance(model_weights, Path):
        weights_arg: str | Path = resolve_path(model_weights, resolved_base_dir)
    else:
        weight_str = str(model_weights)
        candidate = Path(weight_str).expanduser()
        has_path_separator = any(sep in weight_str for sep in {os.sep, os.altsep} if sep)
        if has_path_separator or candidate.exists():
            weights_arg = resolve_path(candidate, resolved_base_dir)
        else:
            weights_arg = weight_str

    if isinstance(weights_arg, Path) and not weights_arg.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_arg}")

    model = YOLO(str(weights_arg))

    model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(project_dir),
        name=run_name,
        auto_augment=auto_augment,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the YOLO segmentation model.")
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
        "--data-config",
        type=Path,
        default=DEFAULT_DATA_CONFIG,
        help="Relative path to the dataset YAML file (resolved against the base dir).",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=DEFAULT_PROJECT_SUBDIR,
        help="Directory where training runs will be stored (relative to the base dir).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=DEFAULT_RUN_NAME,
        help="Name of the training run directory.",
    )
    parser.add_argument(
        "--model-weights",
        type=str,
        default=DEFAULT_MODEL_WEIGHTS,
        help="Initial YOLO weights to start training from.",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument(
        "--auto-augment",
        type=str,
        default=None,
        help="Optional auto augment policy passed to Ultralytics.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete an existing run directory with the same name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(
        base_dir=args.base_dir,
        data_config=args.data_config,
        project_dir=args.project_dir,
        run_name=args.run_name,
        model_weights=args.model_weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        auto_augment=args.auto_augment,
        overwrite=not args.keep_existing,
    )


if __name__ == "__main__":
    main()
