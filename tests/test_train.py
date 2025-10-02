"""Tests for the train module."""

from __future__ import annotations

from pathlib import Path
import sys
import types
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

stub_ultralytics = types.ModuleType("ultralytics")


class _StubYOLO:  # pragma: no cover - simple stub used for import
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def train(self, *args, **kwargs):
        return None


stub_ultralytics.YOLO = _StubYOLO
sys.modules.setdefault("ultralytics", stub_ultralytics)

import train


def test_train_model_resolves_weights_relative_to_base_dir(tmp_path):
    """Ensure string weights names resolve relative to the provided base dir."""

    base_dir = tmp_path
    (base_dir / "datasets").mkdir()
    data_config = base_dir / "datasets" / "data.yaml"
    data_config.write_text("path: datasets\n")

    weights_file = base_dir / "custom.pt"
    weights_file.write_text("dummy weights")

    with patch.object(train, "YOLO") as mock_yolo:
        mock_yolo.return_value.train.return_value = None

        train.train_model(
            base_dir=base_dir,
            data_config=Path("datasets/data.yaml"),
            project_dir=Path("runs/detect"),
            run_name="test-run",
            model_weights="custom.pt",
            epochs=1,
            imgsz=8,
            batch=1,
        )

    assert mock_yolo.call_args is not None
    (weights_arg,), _ = mock_yolo.call_args
    assert Path(weights_arg) == weights_file
