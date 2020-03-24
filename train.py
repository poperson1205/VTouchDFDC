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
    list_metadata_df.append(pd.read_csv(os.path.join(CSV_DIR, 'metadata_%d.csv' % i)).sample(frac=0.1, random_state=i).reset_index(drop=True))
metadata_df = pd.concat(list_metadata_df, ignore_index=True)
metadata_df = metadata_df.loc[metadata_df['manipulated_ratio'] > 0.001]

metadata_fake_df = pd.DataFrame({'image_name': metadata_df['image_name'], 'label': ['FAKE' for i in range(len(metadata_df))]}).sample(frac=1.0, random_state=1).reset_index(drop=True)
metadata_real_df = pd.DataFrame({'image_name': metadata_df['original'], 'label' : ['REAL' for i in range(len(metadata_df))]}).sample(frac=1.0, random_state=2).reset_index(drop=True)
metadata_df = pd.concat([metadata_fake_df, metadata_real_df], ignore_index=True)

gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

image_size = 224
batch_size = 8
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean,std)

def isotropically_resize_image(img, size, resample=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if w > h:
        h = h * size // w
        w = size
    else:
        w = w * size // h
        h = size
    resized = cv2.resize(img, (w, h), interpolation=resample)
    return resized

def make_square_image(img):
    h, w = img.shape[:2]
    size = max(h, w)
    t = 0
    b = size - h
    l = 0
    r = size - w
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)

def load_image_as_tensor(image_path, image_size=224):
    img = cv2.imread(image_path)
    img = isotropically_resize_image(img, image_size)
    img = make_square_image(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return normalize_transform(torch.tensor(img).permute((2, 0, 1)).float().div(255))


from torch.utils.data import Dataset



class VideoDataset(Dataset):
    
    def __init__(self, root_dir, df, image_size=224):
        self.root_dir = root_dir
        self.image_size = image_size
        self.df = df
        # self.df = df.sample(frac=1).reset_index(drop=True)
        
        num_fake_imgs = len(self.df)
        print('# fake images: %d' % num_fake_imgs)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        class_index = 1 if row['label'] == 'FAKE' else 0
        file_name = row['image_name']
        image_tensor = load_image_as_tensor(os.path.join(self.root_dir, file_name), self.image_size)
        return image_tensor, class_index

    def __len__(self):
        return len(self.df)



from torch.utils.data import DataLoader

def create_data_loaders(root_dir, metadata_df, image_size, batch_size, num_workers):
    dataset = VideoDataset(root_dir, metadata_df, image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    return loader

train_loader = create_data_loaders(ROOT_DIR, metadata_df, image_size, batch_size, num_workers=0)



model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=1)
model.cuda()



import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(1):
    
    bce_loss = 0.0
    total_examples = 0

    step_count = 0
    for data in tqdm(train_loader):
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

        print('batch BCE: %.4f' % (batch_bce))

        step_count += 1
        if step_count % 1000 == 0:
            torch.save(model.state_dict(), 'binary_classifier_efficientnet_b7_%d.pth' % step_count)

    bce_loss /= total_examples
    print('Epoch: %3d, train BCE: %.4f' % (epoch+1, bce_loss))


print('Finished Training')

torch.save(model.state_dict(), 'binary_classifier_efficientnet_b7.pth')