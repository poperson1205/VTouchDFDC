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

# Read dataframe
list_metadata_df = []
list_folder_index = range(0, 45)
for i in list_folder_index:
    folder_name = 'dfdc_train_part_%d' % i
    list_metadata_df.append(pd.read_csv('metadata_%d.csv' % i))
metadata_df = pd.concat(list_metadata_df)

gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

image_size = 224
batch_size = 64
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean,std)

def load_image_as_tensor(image_path, image_size=224):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    return normalize_transform(torch.tensor(img).permute((2, 0, 1)).float().div(255))


from torch.utils.data import Dataset



class VideoDataset(Dataset):
    
    def __init__(self, root_dir, df, image_size=224):
        self.root_dir = root_dir
        self.image_size = image_size
        self.df = df
        
        num_fake_imgs = len(df)
        print('# fake images: %d' % num_fake_imgs)

    def __getitem__(self, index):
        is_fake = True if index % 2 == 0 else False
        class_index = 1 if is_fake else 0
        row = self.df.iloc[int(index/2)]
        file_name = row['image_name'] if is_fake else row['original']
        image_tensor = load_image_as_tensor(os.path.join(self.root_dir, file_name), self.image_size)
        return image_tensor, class_index

    def __len__(self):
        return len(self.df)



from torch.utils.data import DataLoader

def create_data_loaders(root_dir, metadata_df, image_size, batch_size, num_workers):
    dataset = VideoDataset(root_dir, metadata_df, image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return loader

train_loader = create_data_loaders(ROOT_DIR, metadata_df, image_size, batch_size, num_workers=0)



model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1)
model.cuda()



import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(5):
    
    bce_loss = 0.0
    total_examples = 0

    for batch_idx, data in enumerate(train_loader):
        batch_size = data[0].shape[0]
        x = data[0].to(gpu)
        y_true = data[1].to(gpu).float()
        
        optimizer.zero_grad()

        y_pred = model(x)
        y_pred = y_pred.squeeze()

        loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
        loss.backward()
        optimizer.step()

        batch_bce = loss.item()
        bce_loss += batch_bce * batch_size
        total_examples += batch_size

        print('Progress: %3d / %3d, batch BCE: %.4f' % (batch_idx+1, len(train_loader), batch_bce))

    bce_loss /= total_examples
    print('Epoch: %3d, train BCE: %.4f' % (epoch+1, bce_loss))


print('Finished Training')

torch.save(model.state_dict(), 'binary_classifier.pth')