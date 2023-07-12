from super_gradients.common.object_names import Models
from super_gradients.training import models
import random
import cv2
import numpy as np
import time
import torch
import timeit

model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco").cuda()
model.eval()

# media_predictions = model.predict("vid.mp4")
# media_predictions.save("vid_output.mp4")
# model = model.to("cuda" if torch.cuda.is_available() else "cpu")
global label_colors
global names



cap = cv2.VideoCapture("blackbox_rain.mp4")
while cap.isOpened() :
    ret, frame = cap.read()
    if ret == True :
        start_t = timeit.default_timer()
        first_frame = True
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        outputs = model.predict(frame, conf=0.4, iou=0.4) # 
        output = outputs[0]
        bboxes = output.prediction.bboxes_xyxy
        confs = output.prediction.confidence
        labels = output.prediction.labels
        class_names = output.class_names

        if first_frame :
            random.seed(0)
            labels = [int(l) for l in list(labels)]
            label_colors = [tuple(random.choices(np.arange(0, 256), k=3)) for i in range(len(class_names))]
            names = [class_names[int(label)] for label in labels]
            first_frame = False

        for idx, bbox in enumerate(bboxes):
            bbox_left = int(bbox[0])
            bbox_top = int(bbox[1])
            bbox_right = int(bbox[2])
            bbox_bot = int(bbox[3])

            text = f"{names[idx]} {confs[idx]:.2f}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            colors = tuple(int(i) for i in label_colors[labels[idx]])
            cv2.rectangle(frame, (bbox_left, bbox_top), (bbox_left + text_w, bbox_top - text_h), colors, -1)
            cv2.putText(frame, text, (bbox_left, bbox_top), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            cv2.rectangle(frame, (bbox_left, bbox_top), (bbox_right, bbox_bot), color=colors, thickness=2)

        terminate_t = timeit.default_timer()
        
        FPS = 1./(terminate_t - start_t )
        print(f"FPS : {FPS:.2f}")


        cv2.imshow("test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()