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
    list_original = []
    list_manipulated_ratio = []
    list_img_min = []
    list_img_max = []
    list_diff_min = []
    list_diff_max = []
    for video_name, attributes in tqdm(metadata.items()):
        if attributes['label'] == 'REAL':
            continue

        video_name_original = attributes['original']

        if not ('face' in attributes):
            continue

        for frame_index, detection_results in attributes['face'].items():
            for box_index, box_result in enumerate(detection_results):
                img_name = '%s-%d-%d.png' % (video_name[:-4], int(frame_index), box_index)
                if not (img_name in files_in_folder):
                    continue

                img_name_original = '%s-%d-%d.png' % (video_name_original[:-4], int(frame_index), box_index)
                if not (img_name_original in files_in_folder):
                    continue

                manipulated_ratio = box_result['manipulated_ratio']
                img_min = box_result['img_min']
                img_max = box_result['img_max']
                diff_min = box_result['diff_min']
                diff_max = box_result['diff_max']

                list_image_name.append('%s/%s' % (folder_name, img_name))
                list_original.append('%s/%s' % (folder_name, img_name_original))
                list_manipulated_ratio.append(manipulated_ratio)
                list_img_min.append(img_min)
                list_img_max.append(img_max)
                list_diff_min.append(diff_min)
                list_diff_max.append(diff_max)

    metadata_df = pd.DataFrame({
        'image_name':list_image_name,
        'original':list_original,
        'manipulated_ratio': list_manipulated_ratio,
        'img_min': list_img_min,
        'img_max': list_img_max,
        'diff_min': list_diff_min,
        'diff_max': list_diff_max
        })
    
    return metadata_df

if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    list_metadata_df = []
    for i in range(0,5):
        dir_name = 'dfdc_train_part_%d' % i
        print(dir_name)
        if os.path.isfile(os.path.join(os.path.join(ROOT_DIR, dir_name), 'metadata.json')):
            metadata_df = get_dataframe(ROOT_DIR, dir_name)
            metadata_df.to_csv(os.path.join(OUTPUT_DIR, 'metadata_%d.csv' % i))
            list_metadata_df.append(metadata_df)

    metadata_df = pd.concat(list_metadata_df)
    metadata_df.to_csv(os.path.join(OUTPUT_DIR, 'metadata.csv'))