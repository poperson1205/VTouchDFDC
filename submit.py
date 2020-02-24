# MODE = 'KAGGLE'
MODE = 'LOCAL'

if MODE == 'LOCAL':
    model_parameter_path = './binary_classifier.pth'
    test_dataset_dir = '../deepfake-detection-challenge/train_sample_videos'
    output_submission_file_path = './submission.csv'
elif MODE == 'KAGGLE':
    model_parameter_path = '/kaggle/input/models/binary_classifier.pth'
    test_dataset_dir = '/kaggle/input/deepfake-detection-challenge/test_videos'
    output_submission_file_path = '/kaggle/working/submission.csv'
    efficientnet_pakage_path = '../input/efficientnetpytorch/EfficientNet-PyTorch'
    sys.path.append(efficientnet_pakage_path)
    os.system('pip install /kaggle/input/mtcnnwheel/mtcnn-0.1.0-py3-none-any.whl')

import os, sys, random
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

from efficientnet_pytorch import EfficientNet

from mtcnn import MTCNN

from PIL import Image
import sys





image_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean,std)
gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
detector = MTCNN()





def predict(video_path):
    default_prediction = 0.5
    capture_image = cv2.VideoCapture(video_path)

    # Get first frame
    ret, cv_img = capture_image.read()
    if ret is False:
        return default_prediction
    img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img)
    if results is None or len(results) == 0:
        return default_prediction

    # Get face
    box = np.array(results[0]['box'])
    box = box.astype(int)
    face_img = cv_img[box[1]:(box[1]+box[3]), box[0]:(box[0]+box[2])]
    if face_img.size == 0:
        return default_prediction

    # Convert image to tensor        
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = cv2.resize(face_img, (image_size, image_size))
    x = torch.tensor(face_img).permute((2, 0, 1))
    x = normalize_transform(x / 255.)
    x = x.unsqueeze(0).cuda()

    with torch.no_grad():
        y = torch.sigmoid(model(x).squeeze()).cpu().numpy()
        return y





model = EfficientNet.from_name('efficientnet-b0', {'num_classes': 1})
checkpoint = torch.load(model_parameter_path, map_location=gpu)
model.load_state_dict(checkpoint)
model.cuda()
model.train(False)

predictions = []

for root, _, filenames in os.walk(test_dataset_dir):
    for count, filename in enumerate(filenames, 1):
        if filename.endswith('.mp4'):
            filepath = os.path.join(root, filename)
            prediction = predict(filepath)
            predictions.append([filename, prediction])
            print('%s: %f (%3d / %3d)' % (filename, prediction, count, len(filenames)))
            
df = pd.DataFrame(data=predictions, columns=['filename', 'label'])
df.sort_values('filename').to_csv(output_submission_file_path, index=False)
print('complete!')