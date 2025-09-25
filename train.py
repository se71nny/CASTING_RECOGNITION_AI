from pathlib import Path
import shutil
from ultralytics import YOLO

def train_model():
    root_dir = Path(__file__).resolve().parent.parent
    project_dir = root_dir / "runs" / "detect"
    name = "train_fixed_aug"
    save_path = project_dir / name

    project_dir.mkdir(parents=True, exist_ok=True)

    if save_path.exists():
        try:
            save_path.resolve().relative_to(project_dir)
        except ValueError as exc:
            raise RuntimeError(f"Refusing to delete path outside {project_dir}: {save_path}") from exc

        try:
            shutil.rmtree(save_path)
        except OSError as exc:
            raise RuntimeError(f"Failed to remove previous run directory: {save_path}") from exc

    data_path = root_dir / "datasets" / "data.yaml"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    model = YOLO("yolov8s-seg.pt")

    model.train(
        data=str(data_path),
        epochs=25,
        imgsz=640,
        batch=12,
        project=str(project_dir),
        name=name,
        auto_augment=None,
    )

if __name__ == "__main__":
    train_model()