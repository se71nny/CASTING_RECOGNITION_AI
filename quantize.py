"""Export Ultralytics YOLO weights to ONNX and quantize them into HAR artifacts."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from ultralytics import YOLO
import torch
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel, SegmentationModel, ClassificationModel
add_safe_globals([DetectionModel, SegmentationModel, ClassificationModel])

DEFAULT_WEIGHTS = Path("runs/detect/train_fixed_aug/weights/best.pt")
DEFAULT_IMAGE_SIZE = 640
DEFAULT_HAILO_ARCH = "hailo8"


class CommandExecutionError(RuntimeError):
    """Raised when an external command used by this script fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a trained YOLO model to ONNX and produce quantized HAR files "
            "ready for compilation with the Hailo toolchain."
        )
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to the trained YOLO weights (.pt).",
    )

    parser.add_argument(
        "--onnx",
        type=str,
        default=None,
        help="Path to an existing ONNX file to reuse instead of exporting again."
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Image size to use during ONNX export (passed to Ultralytics).",
    )
    parser.add_argument(
        "--onnx-output",
        type=Path,
        default=None,
        help="Destination for the exported ONNX file. Defaults to <weights>.onnx.",
    )
    parser.add_argument(
        "--hailo-arch",
        type=str,
        default=DEFAULT_HAILO_ARCH,
        help="Target Hailo architecture used by the Hailo parser and optimizer.",
    )
    parser.add_argument(
        "--reuse-onnx",
        action="store_true",
        help="Skip ONNX re-export if already exists.",
    )
    return parser.parse_args()


def ensure_parent_exists(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def export_to_onnx(weights_path: Path, output_path: Path, imgsz: int, overwrite: bool) -> Path:
    """Export YOLO weights to ONNX."""
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    ensure_parent_exists(output_path)
    if output_path.exists():
        if overwrite:
            print(f"Existing ONNX detected, removing {output_path} before re-export.")
            output_path.unlink()
        else:
            print(f"Reusing existing ONNX file: {output_path}")
            return output_path

    model = YOLO(str(weights_path))
    exported = Path(
        model.export(format="onnx", imgsz=imgsz, opset=13, simplify=True)
    )

    if not exported.exists():
        raise FileNotFoundError(
            f"Ultralytics export failed, no ONNX file produced at {exported}"
        )

    if exported.resolve() != output_path.resolve():
        shutil.move(str(exported), output_path)

    return output_path


def run_command(command: list[str]) -> None:
    """Run a subprocess command safely."""
    print(f"\n[Running] {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise CommandExecutionError(
            f"Command failed: {' '.join(command)}"
        ) from e


def convert_to_har(onnx_path: Path, har_path: Path, arch: str) -> Path:
    """Convert ONNX to HAR using Hailo parser."""
    ensure_parent_exists(har_path)
    command = [
        "hailo", "parser", "onnx",
        str(onnx_path),
        "--har-path", str(har_path),
        "--hw-arch", arch,
        "-y"
    ]
    run_command(command)
    if not har_path.exists():
        raise FileNotFoundError(f"HAR file was not created: {har_path}")
    return har_path


def quantize_har(har_path: Path, quantized_har_path: Path, arch: str) -> Path:
    """Quantize HAR model using Hailo optimize."""
    ensure_parent_exists(quantized_har_path)
    command = [
        "hailo", "optimize",
        str(har_path),
        "--hw-arch", arch,
        "--use-random-calib-set",
        "--output-har-path", str(quantized_har_path)
    ]
    run_command(command)
    if not quantized_har_path.exists():
        raise FileNotFoundError(f"Quantized HAR file not found: {quantized_har_path}")
    return quantized_har_path


def main() -> None:
    args = parse_args()

    # define paths
    weights_path = Path(args.weights)
    default_onnx = weights_path.with_suffix(".onnx")
    onnx_path = Path(args.onnx_output) if args.onnx_output else default_onnx
    base_name = onnx_path.stem
    default_har = onnx_path.parent / f"{base_name}.har"
    quantized_har = onnx_path.parent / f"{base_name}_optimized.har"

    onnx_arg = getattr(args, "onnx", None)

    # export ONNX
    if onnx_arg:
        exported_path = Path(onnx_arg)
        if not exported_path.exists():
            raise FileNotFoundError(f"Provided ONNX file does not exist: {exported_path}")
        print(f"Reusing existing ONNX file: {exported_path}")
    else:
        exported_path = export_to_onnx(weights_path, onnx_path, args.imgsz, overwrite=not args.reuse_onnx)
    print(f"✅ Exported ONNX model: {exported_path}")

    # ONNX → HAR
    har_path = convert_to_har(exported_path, default_har, args.hailo_arch)
    print(f"✅ Created HAR model: {har_path}")

    # HAR → Quantized HAR
    quantized_path = quantize_har(har_path, quantized_har, args.hailo_arch)
    print(f"✅ Quantized HAR model: {quantized_path}")

    print(
        "ℹ️ Quantized HAR ready for Hailo compiler CLI. Run 'hailo compiler' manually "
        "to produce the final HEF file."
    )


if __name__ == "__main__":
    main()
