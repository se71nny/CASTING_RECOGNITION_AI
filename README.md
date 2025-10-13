# CASTING_RECOGNITION_AI

## Quantization

1. Install the required dependencies (choose the onnxruntime build that matches your platform):
   ```bash
   # Linux / Windows
   pip install ultralytics onnxruntime onnxruntime-tools opencv-python

   # Apple Silicon (M1/M2)
   pip install ultralytics onnxruntime-silicon onnxruntime-tools opencv-python
   ```
2. Export the trained weights to ONNX and produce an INT8 model:
   ```bash
   python quantize.py \
       --weights runs/detect/train_fixed_aug/weights/best.pt \
       --calib-images datasets/test/images
   ```
   * `best.onnx` will be created next to the weights file.
   * When `onnxruntime` is not installed the script will still export the ONNX model, but skips INT8 quantization and prints a hint describing the missing dependency.
   * Use `--onnx-output` or `--int8-output` to save the outputs elsewhere.

## Running inference with ONNX/INT8 models

### Batch inference on a directory
```bash
python detect.py --model-path runs/detect/train_fixed_aug/weights/best-int8.onnx
```
Use `--image-dir`, `--project-dir`, or `--run-name` to adjust inputs and outputs. Pass `--no-show` to disable GUI windows.

### Webcam stream
```bash
python camera_stream.py --model-path runs/detect/train_fixed_aug/weights/best-int8.onnx
```
Press `q` to stop the stream. Omit `--model-path` to fall back to the original `best.pt` weights.
