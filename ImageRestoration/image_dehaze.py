from huggingface_hub import from_pretrained_keras
from PIL import Image

import tensorflow as tf
import numpy as np
import requests
import time
# 85CF_SD_20220119_053043.jpg


# url = "https://github.com/sayakpaul/maxim-tf/raw/main/images/Deraining/input/55.png"
# image = Image.open(requests.get(url, stream=True).raw)
# image = np.array(image)
# image = tf.convert_to_tensor(image)
# image = tf.image.resize(image, (256, 256))

# model = from_pretrained_keras("google/maxim-s2-deraining-raindrop")
# st = time.time()
# predictions = model.predict(tf.expand_dims(image, 0))
# print(time.time()-st)

task = "Enhancement"  # @param ["Denoising", "Dehazing_Indoor", "Dehazing_Outdoor", "Deblurring", "Deraining", "Enhancement", "Retouching"]

# 목표 : 기상상황마다 다르게 적용되도록 만들기.

model_handle_map = {
    "Denoising": [
        "https://tfhub.dev/sayakpaul/maxim_s-3_denoising_sidd/1",
        "https://github.com/google-research/maxim/raw/main/maxim/images/Denoising/input/0003_30.png",
    ],
    "Dehazing_Indoor": [
        "https://tfhub.dev/sayakpaul/maxim_s-2_dehazing_sots-indoor/1",
        "https://github.com/google-research/maxim/raw/main/maxim/images/Dehazing/input/0003_0.8_0.2.png",
    ],
    "Dehazing_Outdoor": [
        "https://tfhub.dev/sayakpaul/maxim_s-2_dehazing_sots-outdoor/1",
        "https://github.com/google-research/maxim/raw/main/maxim/images/Dehazing/input/1444_10.png",
    ],
    "Deblurring": [
        "https://tfhub.dev/sayakpaul/maxim_s-3_deblurring_gopro/1",
        "https://github.com/google-research/maxim/raw/main/maxim/images/Deblurring/input/1fromGOPR0950.png",
    ],
    "Deraining": [
        "https://tfhub.dev/sayakpaul/maxim_s-2_deraining_raindrop/1",
        "https://github.com/google-research/maxim/raw/main/maxim/images/Deraining/input/15.png",
    ],
    "Enhancement": [
        "https://tfhub.dev/sayakpaul/maxim_s-2_enhancement_lol/1",
        "https://github.com/google-research/maxim/raw/main/maxim/images/Enhancement/input/a4541-DSC_0040-2.png",
    ],
    "Retouching": [
        "https://tfhub.dev/sayakpaul/maxim_s-2_enhancement_fivek/1",
        "https://github.com/google-research/maxim/raw/main/maxim/images/Enhancement/input/a4541-DSC_0040-2.png",
    ],
}

model_handle = model_handle_map[task]
ckpt = model_handle[0]
print(f"TF-Hub handle: {ckpt}.")

import cv2

import tensorflow as tf
import tensorflow_hub as hub
from create_maxim_model import Model
from maxim.configs import MAXIM_CONFIGS
import matplotlib.pyplot as plt
import time
from PIL import Image
import numpy as np

# image_url = model_handle[1]
# image_path = tf.keras.utils.get_file(origin=image_url)
# Image.open(image_path)


# Since the model was not initialized to take variable-length sizes (None, None, 3),
# we need to be careful about how we are resizing the images.
# From https://www.tensorflow.org/lite/examples/style_transfer/overview#pre-process_the_inputs
def resize_image(image, target_dim):
    # Resize the image so that the shorter dimension becomes `target_dim`.
    shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
    short_dim = min(shape)
    scale = target_dim / short_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, new_shape)

    # Central crop the image.
    image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)

    return image


def process_image(image_path, target_dim=256):
    input_img = np.asarray(Image.open(image_path).convert("RGB"), np.float32) / 255.0
    input_img = tf.expand_dims(input_img, axis=0)
    input_img = resize_image(input_img, target_dim)
    return input_img

def get_model(model_url: str, input_resolution: tuple) -> tf.keras.Model:
    inputs = tf.keras.Input((*input_resolution, 3))
    hub_module = hub.KerasLayer(model_url)

    outputs = hub_module(inputs)

    return tf.keras.Model(inputs, outputs)


# Based on https://github.com/google-research/maxim/blob/main/maxim/run_eval.py
def infer(image_path: str, model: tf.keras.Model, input_resolution=(256, 256)):
    preprocessed_image = process_image(image_path, input_resolution[0])

    preds = model.predict(preprocessed_image)
    if isinstance(preds, list):
        preds = preds[-1]
        if isinstance(preds, list):
            preds = preds[-1]

    preds = np.array(preds[0], np.float32)
    final_pred_image = np.array((np.clip(preds, 0.0, 1.0)).astype(np.float32))
    return final_pred_image


input_resolution = (256, 256)

model = get_model(ckpt, input_resolution)

# img_path="../../summer_yolo/train/haze/85CF_HD_20211018_038934.jpg"
# img_path= "../../sfw_yolo/train/snow/85CF_SD_20220119_052885.jpg" # snow
# img_path = "../test.jpg"
img_path="../night_test.jpg"
org = cv2.imread(img_path)
pred_frame = infer(img_path, model, input_resolution)
if len(pred_frame.shape) > 3 :
    pred_frame = tf.squeeze(pred_frame, axis=0)

else :
    pred_frame = (pred_frame*255).astype(np.uint8)

pred_frame = cv2.cvtColor(pred_frame, cv2.COLOR_RGB2BGR)
pred_frame = cv2.resize(pred_frame, (640, 480))
hcon = cv2.hconcat([cv2.resize(org, (640, 480)),pred_frame])
cv2.imshow("origin(L)/result(R)", hcon)
cv2.waitKey(0)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
cv2.destroyAllWindows()


