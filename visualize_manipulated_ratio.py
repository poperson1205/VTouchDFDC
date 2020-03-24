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

from tqdm import tqdm

ROOT_DIR = '/media/vtouchinc02/database/RawData/deepfake-32frame/'
CSV_DIR = '/media/vtouchinc02/database/RawData/deepfake-32frame-csv/'

# Read dataframe
list_metadata_df = []
list_folder_index = range(0, 50)
for i in list_folder_index:
    list_metadata_df.append(pd.read_csv(os.path.join(CSV_DIR, 'metadata_%d.csv' % i)))
metadata_df = pd.concat(list_metadata_df, ignore_index=True)
metadata_df['manipulated_ratio'].hist(bins=100)
plt.savefig('hist_original.png')
print('len (original): %d' % len(metadata_df))
metadata_df = metadata_df.loc[metadata_df['manipulated_ratio'] > 0.1]
metadata_df['manipulated_ratio'].hist(bins=100)
plt.savefig('hist_filtered.png')
print('len (filtered): %d' % len(metadata_df))
