import cv2
from ultralytics import YOLO

model = YOLO(
    "/home/rtx5070/Desktop/dataset_vat_tu_cong_truong/runs/detect/construction_project/yolov8_train_thuocla_02-2/weights/best.pt"
)

video_path = "/home/rtx5070/Desktop/dataset_vat_tu_cong_truong/7610862-uhd_2160_4096_25fps.mp4"

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30

delay = int(1000 / fps)


while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        source=frame,
        conf=0.35,
        imgsz=640,
        verbose=False
    )

    annotated_frame = results[0].plot()

    cv2.imshow("Smoking Detection", annotated_frame)

    key = cv2.waitKey(delay) & 0xFF

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()