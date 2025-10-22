"""Utilities for exporting YOLO weights to ONNX and compiling them to HEF for Hailo-8."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

from ultralytics import YOLO

DEFAULT_WEIGHTS = Path("runs/detect/train_fixed_aug/weights/best.pt")
DEFAULT_IMAGE_SIZE = 640
DEFAULT_HAILO_ARCH = "hailo8"


class CommandExecutionError(RuntimeError):
    """Raised when an external command used by this script fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a trained YOLO model to ONNX and compile the ONNX file into a "
            "HEF artifact suitable for running on Hailo-8 hardware."
        )
    )
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
        help="Image size to use during ONNX export (passed to Ultralytics).",
    )
    parser.add_argument(
        "--onnx-output",
        type=Path,
        default=None,
        help="Destination for the exported ONNX file. Defaults to <weights>.onnx.",
    )
    parser.add_argument(
        "--hef-output",
        type=Path,
        default=None,
        help="Destination for the generated HEF file. Defaults to <onnx-output>.hef.",
    )
    parser.add_argument(
        "--hailo-compiler",
        type=str,
        default=None,
        help=(
            "Optional compiler command used to translate ONNX to HEF. "
            "Defaults to the value of $HAILO_COMPILER or 'hailo_compiler'."
        ),
    )
    parser.add_argument(
        "--hailo-arch",
        type=str,
        default=DEFAULT_HAILO_ARCH,
        help="Target Hailo architecture supplied to the compiler (default: hailo8).",
    )
    parser.add_argument(
        "--compiler-args",
        nargs=argparse.REMAINDER,
        default=(),
        help=(
            "Additional arguments appended verbatim to the compiler command. "
            "Use '--' to separate script arguments from compiler arguments."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing HEF file instead of aborting.",
    )
    parser.add_argument(
        "--reuse-onnx",
        action="store_true",
        help=(
            "Skip re-exporting if the target ONNX already exists and reuse the current "
            "file instead."
        ),
    )
    return parser.parse_args()


def ensure_parent_exists(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def export_to_onnx(
    weights_path: Path, output_path: Path, imgsz: int, overwrite: bool
) -> Path:
    """Export the given YOLO weights to ONNX and return the resolved path."""
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
        model.export(
            format="onnx",
            imgsz=imgsz,
            opset=13,
            simplify=True,
        )
    )
    if not exported.exists():  # pragma: no cover - defensive guard
        raise FileNotFoundError(
            "Ultralytics export did not produce an ONNX file as expected: "
            f"{exported}"
        )

    if exported.resolve() != output_path.resolve():
        shutil.move(str(exported), output_path)

    return output_path


def _resolve_compiler_command(explicit_command: str | None) -> str:
    if explicit_command:
        return explicit_command
    env_command = os.getenv("HAILO_COMPILER")
    if env_command:
        return env_command
    return "hailo_compiler"


def run_compiler(
    compiler: str,
    onnx_path: Path,
    hef_path: Path,
    arch: str,
    extra_args: Iterable[str],
    overwrite: bool,
) -> Path:
    ensure_parent_exists(hef_path)
    if hef_path.exists():
        if not overwrite:
            raise FileExistsError(f"HEF output already exists: {hef_path}")
        hef_path.unlink()

    command: list[str] = [compiler, str(onnx_path), "-o", str(hef_path)]
    if arch:
        command.extend(["--arch", arch])
    if extra_args:
        command.extend(extra_args)

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:  # pragma: no cover - external dependency missing
        raise CommandExecutionError(
            f"Compiler command not found: {compiler}. Install the Hailo SDK or "
            "use --hailo-compiler to point to the correct executable."
        ) from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover - surfaces compiler errors
        raise CommandExecutionError(
            "Hailo compiler failed. Inspect the output above for details."
        ) from exc

    if not hef_path.exists():  # pragma: no cover - defensive guard
        raise FileNotFoundError(
            "The Hailo compiler completed without producing the expected HEF: "
            f"{hef_path}"
        )

    return hef_path


def main() -> None:
    args = parse_args()

    default_onnx = args.weights.parent / "best.onnx"
    onnx_path = args.onnx_output or default_onnx
    default_hef = onnx_path.parent / "best.hef"
    hef_path = args.hef_output or default_hef

    exported_path = export_to_onnx(
        args.weights,
        onnx_path,
        args.imgsz,
        overwrite=not args.reuse_onnx,
    )
    print(f"Exported ONNX model to {exported_path}")

    compiler_command = _resolve_compiler_command(args.hailo_compiler)
    compiled_path = run_compiler(
        compiler_command,
        exported_path,
        hef_path,
        args.hailo_arch,
        args.compiler_args,
        args.force,
    )
    print(f"Compiled HEF model to {compiled_path}")


if __name__ == "__main__":
    main()
