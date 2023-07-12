task = "Enhancement"  # @param ["Denoising", "Dehazing_Indoor", "Dehazing_Outdoor", "Deblurring", "Deraining", "Enhancement", "Retouching"]

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

# model_handle = model_handle_map[task]
# ckpt = model_handle[0]
# print(f"TF-Hub handle: {ckpt}.")


import tensorflow as tf
import tensorflow_hub as hub
from create_maxim_model import Model
from maxim.configs import MAXIM_CONFIGS
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np


# _MODEL = tf.keras.models.load_model('maxim_s-2_dehazing_sots-outdoor_1')
# _MODEL = tf.keras.models.load_model('maxim_s-2_enhancement_lol_1')
_MODEL = tf.keras.models.load_model('maxim_s-2_deraining_raindrop_1')





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


def init_new_model(input_img):
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
    new_model.set_weights(_MODEL.get_weights())
    return new_model


def infer(image_path: str):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((640, 480))
    preprocessed_image, height, width, height_even, width_even = process_image(image)
    new_model = init_new_model(preprocessed_image)

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

import cv2
img_path="./test_images/rain4.jpg"
# img = cv2.imread(img_path)

# img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
# cv2.imwrite("rain_test.jpg", img)

final_pred_image = infer(img_path)
input_image = np.asarray(Image.open(img_path).convert("RGB"), np.float32) / 255.0

if len(final_pred_image.shape) > 3 :
    final_pred_image = tf.squeeze(final_pred_image, axis=0)

else :
    final_pred_image = (final_pred_image*255).astype(np.uint8)

final_pred_image = cv2.cvtColor(final_pred_image, cv2.COLOR_RGB2BGR)
# pred_frame = cv2.resize(pred_frame, (640, 480))
org = cv2.imread(img_path)
hcon = cv2.hconcat([cv2.resize(org, (final_pred_image.shape[1], final_pred_image.shape[0])),final_pred_image])
print(final_pred_image.shape)
cv2.imshow("Result", hcon)
cv2.waitKey(0)
cv2.destroyAllWindows()