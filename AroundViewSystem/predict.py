import cv2
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

# video
vid_path = "./samples/vid1.mp4"

cap = cv2.VideoCapture(vid_path)

while cap.isOpened() :
    _, frame = cap.read()
    if _ :
        results = model(frame)

        annotated_frame = results[0].plot()

        cv2.imshow("Inference Results", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else :
        break

cap.release()
cv2.destroyAllWindows()