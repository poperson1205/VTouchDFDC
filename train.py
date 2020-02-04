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
metadata_df.head()

print('# REAL: {0}'.format(len(metadata_df[metadata_df.label == 'REAL'])))
print('# FAKE: {0}'.format(len(metadata_df[metadata_df.label == 'FAKE'])))
print('TOTAL: {0}'.format(len(metadata_df)))