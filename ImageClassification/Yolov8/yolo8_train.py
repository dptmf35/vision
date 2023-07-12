from ultralytics import YOLO
# import torch

model = YOLO('yolov8s-cls.yaml')
model = YOLO('yolov8s-cls.pt')

model.train(data='/mnt/tram_dataset/summer_yolo', epochs=15)
