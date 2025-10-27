# Hailo Quantization & Compilation Pipeline

This project converts trained Ultralytics YOLO checkpoints into binaries that can run on a Hailo-8 accelerator. The flow matches the process we verified manually while experimenting inside the repository.

## Stage 1 – `quantize.py`
`quantize.py` orchestrates the model export and quantization steps entirely through Python:

1. **`.pt` → `.onnx`** – Loads the Ultralytics weights and exports them to ONNX. Existing ONNX files can be reused with `--onnx` or `--reuse-onnx`.
2. **`.onnx` → `.har`** – Invokes `hailo parser onnx` to produce a HAR archive understood by the Hailo toolchain.
3. **`.har` → `_optimized.har`** – Runs `hailo optimize` to quantize and optimise the archive for the target architecture.

Every intermediate artifact is stored next to the original weights by default, and you can override the output locations with CLI flags.

## Stage 2 – Hailo Dataflow Compiler
Run the Hailo compiler CLI manually once a quantized HAR is ready. The dedicated guide in
[`docs/hailo_hef_compilation.md`](./hailo_hef_compilation.md) covers the full command and
common flags, but at a high level you will invoke:

```bash
hailo compiler path/to/model_optimized.har \
  --hw-arch hailo8 \
  --output-dir path/to/weights/ \
  --auto-model-script path/to/auto_model_script.py
```

This step generates `model.hef`, which is the binary that runs on the Hailo-8 device.

## Quick recap
* `quantize.py` covers everything up to a quantized HAR.
* The Hailo compiler transforms the quantized HAR into the deployable HEF file.

Together these stages give you a repeatable pathway from a PyTorch checkpoint to production-ready firmware for Hailo hardware.
