from ultralytics import YOLO
import cv2
import numpy as np

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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        masks = results[0].masks  # 세그멘테이션 마스크 정보

        annotated_frame = frame.copy()
        if masks is not None:
            overlay = annotated_frame.copy()
            for seg, cls in zip(masks.data, results[0].boxes.cls):
                mask = seg.cpu().numpy()
                mask = (mask > 0.5).astype(np.uint8)
                label = results[0].names[int(cls)]
                if label == '0':
                    continue  # '0' 클래스(배경)는 건너뜀

                color = class_colors.get(label, (0, 255, 255))  # 지정 외 클래스는 노란색
                # 컬러 마스크 생성
                color_mask = np.zeros_like(annotated_frame, dtype=np.uint8)
                color_mask[mask == 1] = color
                # overlay에 컬러 마스크 합성
                overlay = cv2.addWeighted(overlay, 1, color_mask, alpha, 0)
                # 라벨 텍스트 표시
                x, y = np.where(mask)
                if len(x) > 0 and len(y) > 0:
                    cv2.putText(overlay, label, (int(np.mean(y)), int(np.mean(x))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            annotated_frame = overlay

        cv2.imshow("Webcam Detection (Seg Only, Transparent)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam_detection()