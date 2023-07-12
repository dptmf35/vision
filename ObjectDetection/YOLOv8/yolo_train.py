import ultralytics

from ultralytics import YOLO

model = YOLO('yolov8s.pt')

model.train(data='./tld_sample/bongo_tld.yaml' , epochs=50)