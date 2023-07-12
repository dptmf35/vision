from super_gradients.common.object_names import Models
from super_gradients.training import models
import random
import cv2
import numpy as np
import time
import torch

model = models.get(Models.YOLO_NAS_L, pretrained_weights="coco").cuda()
model.eval()

media_predictions = model.predict_webcam()
# media_predictions.save("output_l.mp4")
media_predictions.show()


