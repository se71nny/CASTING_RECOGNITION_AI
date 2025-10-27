# Manual HEF Compilation Guide

Once a quantized HAR (`*_optimized.har`) file is available, run the Hailo Dataflow Compiler manually to produce the executable HEF binary for the target device.

## Prerequisites
- Hailo SDK installed with access to the `hailo` CLI tools.
- Quantized HAR artifact generated via `quantize.py` or an equivalent pipeline.
- Optional auto model script provided by the Hailo SDK when required by your model.

## Basic command
```bash
hailo compiler /path/to/model_optimized.har \
  --hw-arch hailo8 \
  --output-dir /path/to/output/ \
  --auto-model-script /path/to/auto_model_script.py
```

### Arguments
- `model_optimized.har`: quantized HAR produced by `hailo optimize`.
- `--hw-arch`: target accelerator architecture (for example `hailo8`).
- `--output-dir`: directory where the compiler writes the resulting `model.hef` file.
- `--auto-model-script`: optional helper script supplied by the SDK for certain models; omit if not required.

## Tips
- Add `--name <custom_name>` when you need the HEF to use a different base filename.
- Use `--force` to overwrite an existing HEF (disabled by default to prevent accidental loss).
- Review the compiler log for calibration or validation warnings that may require you to revisit the quantization step.

After the command finishes successfully, the specified output directory will contain the deployable `*.hef` file ready to run on Hailo hardware.
