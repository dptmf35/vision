from ultralytics import YOLO
import random, cv2, glob
from torchvision import models
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

detector = YOLO("/mnt/Desktop/extraman/extraspace/nano/road_sign/best.pt") 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(model_path, class_folder_path):
    model = models.efficientnet_v2_s(pretrained=False)
    # number of classes
    class_folder_dir=glob.glob(class_folder_path)
    num_classes = len(class_folder_dir) # Change to your number of classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

def predict_image(image, model, transform,idx_to_cls):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities=torch.nn.functional.softmax(outputs,dim=1)
        max_prob, predicted = torch.max(probabilities, 1)
        
        predicted_label = idx_to_cls[predicted.item()]
    return max_prob.item(), predicted_label

# custom Model load
restriction_model = load_model('restriction_best.pth', '../restriction/train/*')
instruction_model = load_model('instruction_best.pth', '../instruction/train/*')
caution_model = load_model('caution_best.pth', '../caution/train/*')


transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ImageFolder 데이터셋 로드, class label
# train_dataset = datasets.ImageFolder(root='../instruction/train/', transform=transform_train)
# idx_to_cls = {v:k for k,v in train_dataset.class_to_idx.items()}
# print("idx-to-cls:",idx_to_cls)

# restriction
restriction_idx_to_cls ={0: 'ped-ban', 1: 'passing-ban', 2: 'yield', 3: 'right-turn-ban', 4: 'U-turn-ban', 5: 'stop', 6: 'left-turn-ban', 7: 'parking-stop-ban', 8: 'parking-ban', 9: 'no entering', 10: 'safety distance', 11: 'height limit', 12: 'slow', 13: 'max speed limit'}

# instruction
instruction_idx_to_cls = {0: 'bicycle', 1: 'both-side traffic', 2: 'bus-only', 3: 'car-only', 4: 'child-protection', 5: 'crosswalk', 6: 'detour', 7: 'left-side traffic', 
             8: 'left-turn', 9: 'one-way traffic', 10: 'passing-direction', 11: 'right-side traffic', 12: 'right-turn', 13: 'roundabout', 14: 'straight', 15: 'straight-and-left-turn', 16: 'straight-and-right-turn', 17: 'turn-left-with-no-signal', 18: 'u-turn'}

# caution
caution_idx_to_cls = {0: 'crossroad', 1: 'riverside road', 2: 'sloping road', 3: 'speedbump', 4: 'rockslide', 5: 'rough road', 6: 'under construction', 7: 'slippery road', 8: 'wild animals', 9: 'child protection', 10: 'right-curved-road', 
              11: 'right-road-disappear', 12: 'right-join-road', 13: 'hazard', 14: 'bicycle', 15: 'left-curved-road', 16: 'double-curved-road', 17: 'left-join-road', 18: 'tunnel', 19: 'wind', 20: 'crosswalk'}




imgpath="./test1.jpg"
image = Image.open(imgpath)
if image.mode == 'RGBA':
    image = image.convert('RGB')
cv_image=cv2.imread(imgpath)

sign_color={}

for name in detector.names.values() : 
    sign_color[name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


results = detector(cv_image)

print(detector.names)

boxes = results[0].boxes
for bbox in boxes : 
    cls = int(bbox.cls.cpu().detach().numpy().tolist()[0])
    x1, y1, x2, y2 = bbox.xyxy.cpu().detach().numpy().tolist()[0]
    box = [int(x1), int(y1), int(x2), int(y2)]
    # image crop and classification
    cropimg = image.crop(box)
    cropped_image = cv_image[int(y1):int(y2),int(x1):int(x2)]
    # box class #1 : restriction (2)
    if cls == 2 :
        max_prob, predicted_label = predict_image(cropimg, restriction_model, transform_train,restriction_idx_to_cls)
    # box class #2 : instruction
    elif cls == 3 :
        max_prob, predicted_label = predict_image(cropimg, instruction_model, transform_train,instruction_idx_to_cls)
    # box class #3 : caution
    elif cls == 4 :
        max_prob, predicted_label = predict_image(cropimg, caution_model, transform_train,caution_idx_to_cls)
    else :
        continue

    bbox_conf = bbox.conf.cpu().detach().numpy().tolist()[0]
    name = detector.names[int(cls)]
    # print(x1, y1, x2, y2, conf, sign_cls.names[cls], color[name])
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(cv_image, (x1, y1), (x2, y2), sign_color[name], 2)
    text = f'{name} - {predicted_label}({max_prob:.2f})'
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(cv_image, (x1, y1 - text_height - baseline), (x1 + text_width, y1), sign_color[name], -1)
    cv2.putText(cv_image, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.imshow("result",cv_image)
# cv2.imwrite("cv_image.jpg",cv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
    