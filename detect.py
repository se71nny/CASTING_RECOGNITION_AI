import shutil
import os
from ultralytics import YOLO

def run_inference():
    project_dir = r"C:\Workspace\CASTING_REGNITION_AI\runs\detect"
    name = "predict_fixed_aug"  # 증강 학습 모델 결과 기반 추론
    save_path = os.path.join(project_dir, name)

    # 실행 전 기존 결과 삭제
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    model = YOLO(r"C:\Workspace\CASTING_REGNITION_AI\runs\detect\train_fixed_aug\weights\best.pt")

    test_dir = r"C:\Workspace\CASTING_REGNITION_AI\datasets\test\images"
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    test_images = [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith(image_extensions)
    ]

    if not test_images:
        print("테스트 이미지가 없습니다!")
        return

    # 추론 + 저장
    results = model(test_images, project=project_dir, name=name, save=True)

    # 화면 출력
    for result in results:
        result.show()

if __name__ == "__main__":
    run_inference()