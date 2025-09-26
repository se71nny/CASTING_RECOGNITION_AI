# CASTING_RECOGNITION_AI

## Quantization

1. Install the required dependencies:
   ```bash
   pip install ultralytics onnxruntime onnxruntime-tools opencv-python
   ```
2. Export the trained weights to ONNX and produce an INT8 model:
   ```bash
   python quantize.py \
       --weights runs/detect/train_fixed_aug/weights/best.pt \
       --calib-images datasets/test/images
   ```
   * `best.onnx` will be created next to the weights file.
   * An INT8 calibrated model named `best-int8.onnx` will be saved in the same directory.
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
