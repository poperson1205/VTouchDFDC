import os, sys, random
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

crops_dir = '../deep-faces/faces_224'

gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

metadata_df = pd.read_csv('../deep-faces/metadata.csv')

def load_image_as_tensor(image_path, image_size=224):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    return torch.tensor(img).permute((2, 0, 1)).float().div(255)





from torch.utils.data import Dataset



class VideoDataset(Dataset):
    
    def __init__(self, crops_dir, df, split, image_size=224):
        self.crops_dir = crops_dir
        self.df = df
        self.split = split
        self.image_size = image_size

    def __getitem__(self, index):
        row = self.df.iloc[index]
        file_name = row['videoname'][:-4] + '.jpg'
        class_index = 1 if row['label'] == 'FAKE' else 0
        image_tensor = load_image_as_tensor(os.path.join(crops_dir, file_name), self.image_size)
        return image_tensor, class_index

    def __len__(self):
        return len(self.df)



dataset = VideoDataset(crops_dir, metadata_df, 'train')

for data in dataset:
    plt.imshow(data[0].permute(1, 2, 0))
    plt.pause(1)

del dataset