import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import pickle
import platform
import copy
import math
import warnings
import torch.optim.lr_scheduler as lr_scheduler
import random
import tqdm
import torch
import torch.nn as nn
import plotly.express as px
import cv2
from torch import optim,cuda
from torch.utils.data import DataLoader, TensorDataset, Dataset,sampler
from torchvision.utils import make_grid
from torchvision import models, datasets
from torchvision import transforms as T
from torchvision import transforms
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import interpolate
from torchsummary import summary
from PIL import Image
from functools import partial
from typing import Callable, Optional
from datetime import datetime as dt
from collections import OrderedDict
from collections import Counter
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from random import randint
from glob import glob
from dataclasses import dataclass
from typing import Any, Callable, Optional, List, Sequence, Tuple, Union
import random
import shutil

model = torch.hub.load('hankyul2/EfficientNetV2-pytorch', 'efficientnet_v2_s', pretrained=True, nclass=4)
train_on_gpu = cuda.is_available()
print("train on gpu :", train_on_gpu)

save_file_name = './pt/sfw_day_weather.pt'
checkpoint_path = './pt/sfw_day_weather.pth'


def load_checkpoint(path):
    """Load a PyTorch model checkpoint

    Params
    --------
        path (str): saved model checkpoint. Must start with `model_name-` and end in '.pth'

    Returns
    --------
        None, save the `model` to `path`

    """

    # Get the model name
    model_name = path.split('-')[0]
    # assert (model_name in ['vgg16',  './pt/sfw_day_weather'
    #                        ]), "Path must have the correct model name"

    # Load in checkpoint
    checkpoint = torch.load(path)

    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        # Make sure to set parameters as not trainable
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']

    elif './pt/sfw_day_weather' in model_name :
        model = 'efficientnet_v2_s'
        # Make sure to set parameters as not trainable
        #for param in model.parameters():
            #param.requires_grad = False
        model = checkpoint

    # Load in the state dict
    #model.load_state_dict(checkpoint['state_dict'])

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} total gradient parameters.')

    # Move to gpu
    #if multi_gpu:
        #model = nn.DataParallel(model)

    if train_on_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    # Model basics
    #model.class_to_idx = checkpoint['class_to_idx']
    #model.idx_to_class = checkpoint['idx_to_class']
    #model.epochs = checkpoint['epochs']

    # Optimizer
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer


model.load_state_dict(torch.load('./pt/sfw_day_weather.pt'), strict=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# model, optimizer = load_checkpoint(path=checkpoint_path)




# print(model)


def process_image(image_path):
    """Process an image path into a PyTorch tensor"""

    image = Image.open(image_path)
    # Resize
    img = image.resize((256, 256))

    # Center crop
    width = 256
    height = 256
    new_width = 224
    new_height = 224

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    img = img.crop((left, top, right, bottom))

    # Convert to numpy, transpose color dimension and normalize
    img = np.array(img).transpose((2, 0, 1)) / 256

    # Standardization
    means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    img = img - means
    img = img / stds

    img_tensor = torch.Tensor(img)

    return img_tensor


def predict(image_path, model, topk=5):
    """Make a prediction for an image using a trained model

    Params
    --------
        image_path (str): filename of the image
        model (PyTorch model): trained model for inference
        topk (int): number of top predictions to return

    Returns
        
    """
    real_class = image_path.split('/')[-2]

    # Convert to pytorch tensor
    img_tensor = process_image(image_path)

    # Resize
    if train_on_gpu:
        img_tensor = img_tensor.view(1, 3, 224, 224).cuda()
    else:
        img_tensor = img_tensor.view(1, 3, 224, 224)

    # Set to evaluation
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(img_tensor)
        ps = torch.exp(out)

        # Find the topk predictions
        topk, topclass = ps.topk(topk, dim=1)

        idx_to_class = {0: 'haze', 1: 'normal', 2: 'rain', 3: 'snow'}

        # Extract the actual classes and probabilities
        top_classes = [
            idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]
        ]
        top_p = topk.cpu().numpy()[0]

        return img_tensor.cpu().squeeze(), top_p, top_classes, real_class



img, top_p, top_classes, real_class = predict('./testimg.jpg', model,topk=4)

print(top_p, top_classes, real_class)

# print("-" * 20)

# test_path = "../sfw/Test/rain/"

# rain_test_path = os.listdir(test_path)[:10]
# print(rain_test_path)
# random.shuffle(rain_test_path)

# for r in rain_test_path :
#     img, top_p, top_classes, real_class = predict(test_path + r, model,topk=4)

#     print(top_p, top_classes, "real class : rain")

# print("-" * 20)


# test_path = "../sfw/Test/normal/"

# normal_test_path = os.listdir(test_path)[:10]
# random.shuffle(normal_test_path)

# for r in normal_test_path :
#     img, top_p, top_classes, real_class = predict(test_path + r, model,topk=4)

#     print(top_p, top_classes, "real class : normal")


# print("-" * 20)

# test_path = "../sfw/Test/haze/"

# haze_test_path = os.listdir(test_path)[:10]
# random.shuffle(haze_test_path)

# for r in haze_test_path :
#     img, top_p, top_classes, real_class = predict(test_path + r, model,topk=4)

#     print(top_p, top_classes, "real class : haze")


# print("-" * 20)


# test_path = "../sfw/Test/snow/"

# snow_test_path = os.listdir(test_path)[:10]
# random.shuffle(snow_test_path)

# for r in snow_test_path :
#     img, top_p, top_classes, real_class = predict(test_path + r, model,topk=4)

#     print(top_p, top_classes, "real class : snow")


# model = torch.hub.load('hankyul2/EfficientNetV2-pytorch', 
#                        'efficientnet_v2_s', pretrained=True, nclass=4)

# model.load_state_dict(torch.load('./weather_v2.pt'))

# class_names = {"0" : "haze",
#               "1" : "normal",
#               "2" : "rain",
#               "3" : "snow"}



# sample_img = Image.open("./testimg.jpg")
# tfms = transforms.Compose([
#                     transforms.Resize((224, 224)),
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                                         std=(0.229, 0.224, 0.225))
#                 ])
# img = tfms(sample_img).unsqueeze(0)
# with torch.no_grad():
#     logits = model(img)
# preds = torch.topk(logits, k=4).indices.squeeze(0).tolist()
# print('-----')
# for idx in preds:
#     label = class_names[str(idx)]
#     prob = torch.softmax(logits, dim=1)[0, idx].item()
#     print('{:<75} ({:.1f}%)'.format(label, prob*100))


cap = cv2.VideoCapture("./blackbox_rain.mp4")

print("--")
while cap.isOpened() :
    ret, frame = cap.read()
    if ret :
        color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_coverted)


        img = pil_image.resize((256, 256))

        # Center crop
        width = 256
        height = 256
        new_width = 224
        new_height = 224

        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        img = img.crop((left, top, right, bottom))

        # Convert to numpy, transpose color dimension and normalize
        img = np.array(img).transpose((2, 0, 1)) / 256

        # Standardization
        means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

        img = img - means
        img = img / stds

        img_tensor = torch.Tensor(img)


        # Resize
        if train_on_gpu:
            img_tensor = img_tensor.view(1, 3, 224, 224).cuda()
        else:
            img_tensor = img_tensor.view(1, 3, 224, 224)

        # Set to evaluation
        with torch.no_grad():
            model.eval()
            # Model outputs log probabilities
            out = model(img_tensor)
            ps = torch.exp(out)

            # Find the topk predictions
            topk, topclass = ps.topk(4, dim=1)

            idx_to_class = {0: 'haze', 1: 'normal', 2: 'rain', 3: 'snow'}

            # Extract the actual classes and probabilities
            top_classes = [
                idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]
            ]
            top_p = topk.cpu().numpy()[0]
            
            print(top_classes, top_p)

# # sample_img = Image.open("./frame.jpg")
# # Preprocess image
#         tfms = transforms.Compose([
#                             transforms.Resize((224, 224)),
#                             transforms.ToTensor(),
#                             transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                                                 std=(0.229, 0.224, 0.225))
#                         ])
#         img = tfms(pil_image).unsqueeze(0)
# # plt.imshow(sample_img)


# # model = EfficientNet.from_pretrained(model_name, num_classes=4)

#         with torch.no_grad():
#             logits = model(img)
#         preds = torch.topk(logits, k=4).indices.squeeze(0).tolist()
#         print('-----')
#         for idx in preds:
#             label = class_names[str(idx)]
#             prob = torch.softmax(logits, dim=1)[0, idx].item()
#             print('{:<75} ({:.1f}%)'.format(label, prob*100))

        # image = cv2.imread("./frame.jpg")
        # text = f'Weather : {class_names[str(preds[0])]} --- {torch.softmax(logits, dim=1)[0, int(preds[0])].item()*100:.2f}%'
        # print(text)
        
        # cv2.putText(frame, text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
        # cv2.imshow("result", frame )
        # cv2.waitKey(2)

# cv2.destroyAllWindows()