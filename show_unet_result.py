import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import cv2
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
from unet import UNet

def sigmoid(X):
   return 1/(1+np.exp(-X))

image_size = 224
batch_size = 16
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean,std)

DIR = '/media/vtouchinc02/database/RawData/deepfake-32frame/dfdc_train_part_1'

# Load model
def convert_image_as_tensor(img, image_size=224):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    return normalize_transform(torch.tensor(img).permute((2, 0, 1)).float().div(255))

gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = UNet(n_channels=3, n_classes=1)
checkpoint = torch.load('binary_classifier_unet.pth', map_location=gpu)
model.load_state_dict(checkpoint)
model.cuda()
model.train(False)

meta_path = os.path.join(DIR, 'metadata.json')
with open(meta_path) as f:
    meta_json = json.load(f)
    plt.ion()
    for video_name, att in meta_json.items():
        if video_name != 'ssuteaoloo.mp4':
            continue

        if not ('original' in att):
            continue
        if att['original'] is None:
            continue

        video_name_original = att['original']
        att_original = meta_json[video_name_original]

        if not ('face' in att_original):
            print('No face detected: %s' % video_name_original)
        
        print('%s --> %s' % (att['original'], video_name))

        face_results = att_original['face']
        good_count = 0
        bad_count = 0
        for frame_index, frame_result in face_results.items():
            # print(frame_index)
            for box_index, box_result in enumerate(frame_result):
                # print('Score: %f' % box_result['score'])

                image_name = '%s-%s-%d.png' % (video_name[:-4], frame_index, box_index)
                img = cv2.imread(os.path.join(DIR, image_name))
                
                cv2.imshow('fake', img)

                # cv2.imshow('original', cv2.resize(img_original, None, None, 0.3, 0.3))
                # cv2.imshow('fake', cv2.resize(img, None, None, 0.3, 0.3))
                # cv2.imshow('diff', cv2.resize(img_diff, None, None, 0.3, 0.3))

                img_tensor = convert_image_as_tensor(img).unsqueeze(0).to(gpu)
                img_output = model(img_tensor)[0].data.cpu().float().numpy()
                img_output = img_output.transpose(1, 2, 0)
                img_output = sigmoid(img_output)
                # img_output = cv2.normalize(img_output, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
                cv2.imshow('fake output', img_output)


                image_name_original = '%s-%s-%d.png' % (video_name_original[:-4], frame_index, box_index)
                img_original = cv2.imread(os.path.join(DIR, image_name_original))
                cv2.imshow('original', img_original)

                img_tensor_original = convert_image_as_tensor(img_original).unsqueeze(0).to(gpu)
                img_output_original = model(img_tensor_original)[0].data.cpu().float().numpy()
                img_output_original = img_output_original.transpose(1, 2, 0)
                img_output_original = sigmoid(img_output_original)
                # img_output = cv2.normalize(img_output, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
                cv2.imshow('original output', img_output_original)
                
                
                img_diff =  cv2.absdiff(img, img_original)
                img_diff = cv2.cvtColor(img_diff, cv2.COLOR_RGB2GRAY)
                img_diff = cv2.normalize(img_diff, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                cv2.imshow('diff', img_diff)


                original_prediction = np.sum(img_output_original)
                fake_prediction = np.sum(img_output)
                # print('ORIGINAL: %f' % original_prediction)
                # print('FAKE: %f' % fake_prediction)

                if original_prediction < fake_prediction:
                    good_count += 1
                else:
                    bad_count += 1

                

                cv2.waitKey()
        print('GOOD COUNT: %d' % good_count)
        print('BAD COUNT: %d' % bad_count)
            