import os, sys, random
import numpy as np
import pandas as pd
import json
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize

from tqdm import tqdm

import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet

ROOT_DIR = '/media/vtouchinc02/database/RawData/deepfake-32frame/'
OUTPUT_DIR = '/media/vtouchinc02/database/RawData/deepfake-32frame-csv'

def get_dataframe(root_dir, folder_name):
    folder_path = os.path.join(root_dir, folder_name)
    files_in_folder = os.listdir(folder_path)

    with open(os.path.join(folder_path, 'metadata.json'), 'r') as fp:
        metadata = json.load(fp)

    list_image_name = []
    for video_name, attributes in tqdm(metadata.items()):
        if attributes['label'] == 'FAKE':
            continue

        if not ('face' in attributes):
            continue

        for frame_index, detection_results in attributes['face'].items():
            for face_index in range(len(detection_results)):
                img_name = '%s-%d-%d.png' % (video_name[:-4], int(frame_index), face_index)
                if not (img_name in files_in_folder):
                    continue
                
                list_image_name.append('%s/%s' % (folder_name, img_name))
                
    metadata_df = pd.DataFrame({
        'image_name':list_image_name
        })
    
    return metadata_df

list_metadata_df = []
for i in range(0,50):
    dir_name = 'dfdc_train_part_%d' % i
    print(dir_name)
    if os.path.isfile(os.path.join(os.path.join(ROOT_DIR, dir_name), 'metadata.json')):
        metadata_df = get_dataframe(ROOT_DIR, dir_name)
        metadata_df.to_csv(os.path.join(OUTPUT_DIR, 'metadata_real_%d.csv' % i))
        list_metadata_df.append(metadata_df)

metadata_df = pd.concat(list_metadata_df)
metadata_df.to_csv(os.path.join(OUTPUT_DIR, 'metadata_real.csv'))