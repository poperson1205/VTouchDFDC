import os, sys, random
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(gpu)