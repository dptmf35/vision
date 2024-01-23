import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

prototxt_path = './model/MobileNetSSD_deploy.prototxt.txt'
model_path = './model/MobileNetSSD_deploy.caffemodel'
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", 
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
    
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
LABEL_COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

cap = cv2.VideoCapture(0) 

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened() :
        ret, frame = cap.read()
        if not ret : 
            break
        start = time.time()
        (h, w) = frame.shape[:2]
        resized = cv2.resize(frame, (300, 300))
        blob = cv2.dnn.blobFromImage(resized, 0.007843, (300, 300), 127.5)

        net.setInput(blob)
        detections = net.forward()

        vis = frame.copy()
        conf = 0.2

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "person" and confidence > conf:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(vis, (startX, startY), (endX, endY), LABEL_COLORS[idx], 1)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(vis, "{} : {:.2f}%".format(CLASSES[idx], confidence * 100), (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, LABEL_COLORS[idx], 2)

        image = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = face_detection.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                if y < 0 : 
                    y = 0
                if x < 0 :
                    x = 0
                ROI = image[y:y+h, x:x+w]
                blur = cv2.GaussianBlur(ROI, (99, 99), 0)
                image[y:y+h, x:x+w] = blur


        cv2.imshow("test", image)
        key = cv2.waitKey(1)
    #     print(f"Processing Time : {time.time() - start:.2f}")
        if key == ord('q') or key == 27:
            break

cap.release()
cv2.destroyAllWindows()