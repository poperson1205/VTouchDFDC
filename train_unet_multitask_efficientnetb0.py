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
from unet_multitask_efficientnetb0 import UNetMultiTaskEfficientNetB0

from tqdm import tqdm

ROOT_DIR = '/media/vtouchinc02/database/RawData/deepfake-32frame/'
CSV_DIR = '/media/vtouchinc02/database/RawData/deepfake-32frame-csv/'

# Read dataframe
list_metadata_df = []
list_folder_index = range(1, 50)
for i in list_folder_index:
    list_metadata_df.append(pd.read_csv(os.path.join(CSV_DIR, 'metadata_%d.csv' % i)))
metadata_df = pd.concat(list_metadata_df)

list_metadata_eval_df = []
list_folder_eval_index = range(0, 1)
for i in list_folder_eval_index:
    list_metadata_eval_df.append(pd.read_csv(os.path.join(CSV_DIR, 'metadata_%d.csv' % i)))
metadata_eval_df = pd.concat(list_metadata_eval_df)

gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

image_size = 224
batch_size = 32
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean,std)

def load_image_as_tensor(image_path, image_size=224):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    return normalize_transform(torch.tensor(img).permute((2, 0, 1)).float().div(255))

def load_image_and_mask_as_tensor(image_path, original_path, image_size=224):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    
    original = cv2.imread(original_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = cv2.resize(original, (image_size, image_size))
    mask = cv2.absdiff(img, original)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    mask = cv2.normalize(mask, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    mask = cv2.resize(mask, (image_size//2, image_size//2))

    return normalize_transform(torch.tensor(img).permute((2, 0, 1)).float().div(255)), torch.tensor(mask)

from torch.utils.data import Dataset



class VideoDataset(Dataset):
    
    def __init__(self, root_dir, df, image_size=224, frac=0.1):
        self.root_dir = root_dir
        self.image_size = image_size
        self.df1 = df.sample(frac=frac).reset_index(drop=True)
        self.df2 = df.sample(frac=frac).reset_index(drop=True)
        
        num_fake_imgs = len(self.df1)
        print('# fake images: %d' % num_fake_imgs)

    def __getitem__(self, index):
        row1 = self.df1.iloc[int(index/2)]
        row2 = self.df2.iloc[int(index/2)]
        # FAKE
        if index % 2 == 0:
            image_name = row1['image_name']
            original_name = row1['original']
            image_path = os.path.join(self.root_dir, image_name)
            original_path = os.path.join(self.root_dir, original_name)
            image_tensor, mask_tensor = load_image_and_mask_as_tensor(image_path, original_path, self.image_size)
            return image_tensor, mask_tensor.unsqueeze(0), 1
        #REAL
        else:
            original_name = row2['original']
            original_tensor = load_image_as_tensor(os.path.join(self.root_dir, original_name), self.image_size)
            return original_tensor, torch.zeros(size=(1, self.image_size//2, self.image_size//2)), 0

    def __len__(self):
        return len(self.df1) * 2



from torch.utils.data import DataLoader

def create_data_loaders(root_dir, metadata_df, image_size, batch_size, num_workers, frac=0.1):
    dataset = VideoDataset(root_dir, metadata_df, image_size, frac=frac)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return loader

train_loader = create_data_loaders(ROOT_DIR, metadata_df, image_size, batch_size, num_workers=0, frac=0.1)
eval_loader = create_data_loaders(ROOT_DIR, metadata_eval_df, image_size, batch_size//4, num_workers=0, frac=0.03)



model = UNetMultiTaskEfficientNetB0(n_channels=3, n_classes=1)
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
        mask_gt = data[1].to(gpu)
        y_gt = data[2].to(gpu).float()
        
        optimizer.zero_grad()

        mask_pred, y_pred = model(x)
        y_pred = y_pred.squeeze()

        mask_loss = F.binary_cross_entropy_with_logits(mask_pred, mask_gt)
        class_loss = F.binary_cross_entropy_with_logits(y_pred, y_gt)
        loss = mask_loss + class_loss
        loss.backward()
        optimizer.step()

        batch_bce = loss.item()
        bce_loss += batch_bce * batch_size
        total_examples += batch_size

        print('batch BCE: %.4f (class: %.4f, mask: %.4f)' % (batch_bce, class_loss.item(), mask_loss.item()))

        step_count += 1
        if step_count % 5000 == 0:
            torch.save(model.state_dict(), 'binary_classifier_unet_multitask_efficientnet_b0_%d.pth' % step_count)

            eval_mask_loss = 0.0
            eval_class_loss = 0.0
            eval_loss = 0.0
            eval_total_examples = 0
            
            for data in eval_loader:
                batch_size = data[0].shape[0]
                x = data[0].to(gpu)
                mask_gt = data[1].to(gpu)
                y_gt = data[2].to(gpu).float()
                
                mask_pred, y_pred = model(x)
                y_pred = y_pred.squeeze()
                
                mask_loss = F.binary_cross_entropy_with_logits(mask_pred, mask_gt)
                class_loss = F.binary_cross_entropy_with_logits(y_pred, y_gt)
                loss = mask_loss + class_loss
                
                eval_mask_loss += mask_loss.item() * batch_size
                eval_class_loss += class_loss.item() * batch_size
                eval_loss += loss.item() * batch_size
                eval_total_examples += batch_size
            
            print('eval BCE: %.4f (class: %.4f, mask: %.4f)' % (eval_loss / eval_total_examples, eval_class_loss / eval_total_examples, eval_mask_loss/ eval_total_examples))

    bce_loss /= total_examples
    print('Epoch: %3d, train BCE: %.4f' % (epoch+1, bce_loss))

print('Finished Training')

torch.save(model.state_dict(), 'binary_classifier_unet_multitask_efficientnet_b0.pth')