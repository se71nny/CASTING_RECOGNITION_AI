"""Run YOLO segmentation on a webcam stream."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_MODEL = Path("runs/detect/train_fixed_aug/weights/best.pt")


def refine_segmentation_mask(binary_mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Return a shrunken mask that better hugs the underlying object."""

    mask_255 = (binary_mask > 0).astype(np.uint8) * 255

    cleaned = cv2.morphologyEx(mask_255, cv2.MORPH_OPEN, kernel)
    refined = cv2.erode(cleaned, kernel, iterations=1)
    return (refined > 0).astype(np.uint8)


def load_model(model_path: Path) -> YOLO:
    if model_path.suffix.lower() == ".onnx":
        return YOLO(str(model_path))
    return YOLO(str(model_path))


def webcam_detection(model_path: Path = DEFAULT_MODEL) -> None:
    model = load_model(model_path)
    cap = cv2.VideoCapture(0)

    alpha = 0.4

    class_colors = {
        "cast1": (0, 0, 255),
        "cast2": (0, 255, 0),
        "cast3": (255, 0, 0),
    }

    refine_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        frame_result = results[0]
        masks = frame_result.masks

        annotated_frame = frame.copy()
        if masks is not None:
            overlay = annotated_frame.copy()
            labels_to_draw = []

            for seg, cls in zip(masks.data, frame_result.boxes.cls):
                mask = seg.cpu().numpy()
                binary_mask = (mask > 0.5).astype(np.uint8)
                refined_mask = refine_segmentation_mask(binary_mask, refine_kernel)

                if not np.any(refined_mask):
                    continue

                if int(refined_mask.sum()) < 30:
                    continue

                label = frame_result.names[int(cls)]
                if label == "0":
                    continue

                color = class_colors.get(label, (0, 255, 255))
                overlay[refined_mask == 1] = color

                ys, xs = np.nonzero(refined_mask)
                if len(xs) and len(ys):
                    labels_to_draw.append((label, (int(xs.mean()), int(ys.mean()))))

            annotated_frame = cv2.addWeighted(annotated_frame, 1 - alpha, overlay, alpha, 0)

            for label, position in labels_to_draw:
                cv2.putText(
                    annotated_frame,
                    label,
                    position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        cv2.imshow("Webcam Detection (Seg Only, Transparent)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run webcam inference using a YOLO model.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to a YOLO model (.pt or .onnx).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    webcam_detection(model_path=args.model_path)


if __name__ == "__main__":
    main()
