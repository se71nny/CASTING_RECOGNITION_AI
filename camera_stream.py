from ultralytics import YOLO
import cv2
import numpy as np


def refine_segmentation_mask(binary_mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Return a shrunken mask that better hugs the underlying object."""

    mask_255 = (binary_mask > 0).astype(np.uint8) * 255

    # 잡음 제거를 위해 작은 구멍과 돌출부를 없애고
    cleaned = cv2.morphologyEx(mask_255, cv2.MORPH_OPEN, kernel)
    # 객체 안쪽으로 한 번 더 수축시켜 외곽 번짐을 완화합니다.
    refined = cv2.erode(cleaned, kernel, iterations=1)
    return (refined > 0).astype(np.uint8)


def webcam_detection():
    model = YOLO("C:/Workspace/CASTING_REGNITION_AI/runs/detect/train_fixed_aug/weights/best.pt")
    cap = cv2.VideoCapture(0)

    alpha = 0.4  # 투명도 (0: 완전 투명, 1: 완전 불투명)

    # 클래스별 색상 지정 (cast1: Red, cast2: Green, cast3: Blue)
    class_colors = {
        'cast1': (0, 0, 255),   # Red (BGR)
        'cast2': (0, 255, 0),   # Green (BGR)
        'cast3': (255, 0, 0),   # Blue (BGR)
    }

    # 마스크 범위를 줄이기 위한 타원형 커널 (3x3) 정의
    refine_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        frame_result = results[0]
        masks = frame_result.masks  # 세그멘테이션 마스크 정보

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

                # 너무 작은 영역은 노이즈로 간주하고 건너뜀
                if int(refined_mask.sum()) < 30:
                    continue

                label = frame_result.names[int(cls)]
                if label == '0':
                    continue  # '0' 클래스(배경)는 건너뜁니다.

                color = class_colors.get(label, (0, 255, 255))  # 지정 외 클래스는 노란색
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

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam_detection()
