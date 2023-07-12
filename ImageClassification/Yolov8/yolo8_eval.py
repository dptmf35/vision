from ultralytics import YOLO

# Load a model
model = YOLO('yolov8s-cls.pt')  # load an official model

model = YOLO('/mnt/tram_dataset/source/runs/classify/train16/weights/best.pt')  # load a custom model

# Predict with the model

path = '/mnt/tram_dataset/sfw/Valid/normal/85CF_ND_20210915_012245.jpg'
results = model(path)
# print('--> answer: normal ')

# for result in results:
#     probs = list(result.probs)
#     classes = result.names
#     print(probs)

# results = model.predict(path)  # predict on an image
# print(results)

# path = '/mnt/tram_dataset/sfw/Valid/haze/85CF_HD_20211018_038972.jpg'
# model(path)
# # results = model.predict(path)  # predict on an image
# print('--> answer: haze ')


# path = '/mnt/tram_dataset/sfw/Valid/snow/85CF_SD_20220117_049243.jpg'
# model(path)
# # results = model.predict(path)  # predict on an image
# print('--> answer: snow ')


# path = '/mnt/tram_dataset/sfw/Valid/rain/85CF_RD_20210917_024348.jpg'
# results = model(path)
# print('--> answer: rain ')

import cv2

cap = cv2.VideoCapture("./blackbox_rain.mp4")

while cap.isOpened() :
    ret, frame = cap.read()
    results = model(frame)
    cv2.imshow("frame", frame)
    for result in results:
        probs = list(result.probs)
        classes = result.names
        print("probs :",probs, "classes :", classes)
        highest_prob = max(probs)
        highest_prob_index = probs.index(highest_prob)

        print(f"Class: {classes[highest_prob_index]}, prob : {highest_prob:.2f}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# results = model.predict(path)  # predict on an image
cv2.destroyAllWindows()