# import ultralytics
import cv2
from ultralytics import YOLO

# model = YOLO('yolov8s.pt')


model = YOLO('./runs/detect/train5/weights/best.pt')

results = model('./test.jpg', ) # conf=0.2

plots = results[0].plot()
cv2.imshow("plot", plots)
cv2.waitKey(0)
cv2.destroyAllWindows()