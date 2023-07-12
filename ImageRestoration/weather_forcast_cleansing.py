from ultralytics import YOLO


model = YOLO('yolov8s-cls.pt')  # load an official model
model = YOLO('/mnt/tram_dataset/source/runs/classify/train14/weights/best.pt')  # load a custom model

import cv2
import tensorflow as tf
import tensorflow_hub as hub
from create_maxim_model import Model
from maxim.configs import MAXIM_CONFIGS
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np

global _MODEL
# global _MODEL
MODEL_dehaze = tf.keras.models.load_model('maxim_s-2_dehazing_sots-outdoor_1')
MODEL_derain = tf.keras.models.load_model('maxim_s-2_deraining_raindrop_1')

weather_class_names = {0: 'haze', 1: 'normal', 2: 'rain', 3: 'snow'}

def mod_padding_symmetric(image, factor=64):
    """Padding the image to be divided by factor."""
    height, width = image.shape[0], image.shape[1]
    height_pad, width_pad = ((height + factor) // factor) * factor, (
        (width + factor) // factor
    ) * factor
    padh = height_pad - height if height % factor != 0 else 0
    padw = width_pad - width if width % factor != 0 else 0
    image = tf.pad(
        image, [(padh // 2, padh // 2), (padw // 2, padw // 2), (0, 0)], mode="REFLECT"
    )
    return image


def make_shape_even(image):
    """Pad the image to have even shapes."""
    height, width = image.shape[0], image.shape[1]
    padh = 1 if height % 2 != 0 else 0
    padw = 1 if width % 2 != 0 else 0
    image = tf.pad(image, [(0, padh), (0, padw), (0, 0)], mode="REFLECT")
    return image


def process_image(image: Image):
    input_img = np.asarray(image) / 255.0
    height, width = input_img.shape[0], input_img.shape[1]

    # Padding images to have even shapes
    input_img = make_shape_even(input_img)
    height_even, width_even = input_img.shape[0], input_img.shape[1]

    # padding images to be multiplies of 64
    input_img = mod_padding_symmetric(input_img, factor=64)
    input_img = tf.expand_dims(input_img, axis=0)
    return input_img, height, width, height_even, width_even


def init_new_model(input_img, status):
    # print(MAXIM_CONFIGS)
    variant = 'S-2'
    configs = MAXIM_CONFIGS.get(variant)
    configs.update(
        {
            "variant": "S-2",
            "dropout_rate": 0.0,
            "num_outputs": 3,
            "use_bias": True,
            "num_supervision_scales": 3,
        }
    )  # From https://github.com/google-research/maxim/blob/main/maxim/run_eval.py#L45-#L61
    configs.update({"input_resolution": (input_img.shape[1], input_img.shape[2])})
    new_model = Model(**configs)
    if status == 'rain' :
        _MODEL= MODEL_derain 
        
    elif status == 'haze' : 
        _MODEL= MODEL_dehaze
    else :
        print('---')
        print(status)
    new_model.set_weights(_MODEL.get_weights())
    return new_model


def infer(frame,status):
    frame_cvt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_cvt)
    image = pil_img.convert("RGB")
    preprocessed_image, height, width, height_even, width_even = process_image(image)
    new_model = init_new_model(preprocessed_image, status)

    preds = new_model.predict(preprocessed_image)
    if isinstance(preds, list):
        preds = preds[-1]
        if isinstance(preds, list):
            preds = preds[-1]

    preds = np.array(preds[0], np.float32)

    new_height, new_width = preds.shape[0], preds.shape[1]
    h_start = new_height // 2 - height_even // 2
    h_end = h_start + height
    w_start = new_width // 2 - width_even // 2
    w_end = w_start + width
    preds = preds[h_start:h_end, w_start:w_end, :]

    return np.array(np.clip(preds, 0.0, 1.0))

def weather_predict(image) :
    results = model(image)
    for result in results:
        probs = list(result.probs)
        classes = result.names

        highest_prob = max(probs)
        highest_prob_index = probs.index(highest_prob)

        text = f"class: {classes[highest_prob_index]}({highest_prob * 100:.2f}%)"

    return classes[highest_prob_index], text



cap = cv2.VideoCapture("./rail_rain(8mm)_narrow.mp4")
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# out = cv2.VideoWriter('result.mp4', fourcc, 15.0, (640, 480))

while cap.isOpened() :
    ret, frame = cap.read()
    if not ret :
        break
    frame = cv2.resize(frame, (640, 480))
    weather, text = weather_predict(frame)
    print(text)
    if weather == 'haze' :
        _MODEL = tf.keras.models.load_model('maxim_s-2_dehazing_sots-outdoor_1')
    elif weather == 'rain' :
        _MODEL = tf.keras.models.load_model('maxim_s-2_deraining_raindrop_1')
    else : 
        pass
    pred_frame = infer(frame, status=weather)
    pred_frame = (pred_frame*255).astype(np.uint8)
    # if len(pred_frame.shape) > 3 :
    # pred_frame = tf.squeeze(pred_frame, axis=0)
    pred_frame = cv2.cvtColor(pred_frame, cv2.COLOR_RGB2BGR)
    # hcon = cv2.hconcat([cv2.resize(frame, (640, 480)),pred_frame])
    cv2.imshow("origin(L)/result(R)", pred_frame)
    # out.write(pred_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# cv2.destroyAllWindows()
# out.release()
cap.release()



