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
from siamese import Siamese

from tqdm import tqdm

ROOT_DIR = '/media/vtouchinc02/database/RawData/deepfake-32frame/'
CSV_DIR = '/media/vtouchinc02/database/RawData/deepfake-32frame-csv/'

# Read dataframe
list_metadata_df = []
list_metadata_real_df = []
list_folder_index = range(2, 48)
for i in list_folder_index:
    list_metadata_df.append(pd.read_csv(os.path.join(CSV_DIR, 'metadata_%d.csv' % i)))
    list_metadata_real_df.append(pd.read_csv(os.path.join(CSV_DIR, 'metadata_real_%d.csv' % i)))
metadata_df = pd.concat(list_metadata_df).sample(frac=0.05).reset_index(drop=True)
metadata_real_df = pd.concat(list_metadata_real_df).sample(frac=0.3).reset_index(drop=True)

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


from torch.utils.data import Dataset



class VideoDataset(Dataset):
    
    def __init__(self, root_dir, df_fake, df_real, image_size=224):
        self.root_dir = root_dir
        self.image_size = image_size
        self.df_fake = df_fake
        self.df_real = df_real
        self.df_real_shuffled = df_real.sample(frac=1).reset_index(drop=True)

        num_fake_imgs = len(self.df_fake)
        num_real_imgs = len(self.df_real)
        print('# fake images: %d' % num_fake_imgs)
        print('# real images: %d' % num_real_imgs)

    def __getitem__(self, index):
        real_real_pair = True if index % 2 == 0 else False
        class_index = 1 if real_real_pair else 0
        if real_real_pair:
            row1 = self.df_real.iloc[int(index/2)]
            row2 = self.df_real_shuffled.iloc[int(index/2)]
            name1 = row1['image_name']
            name2 = row2['image_name']
        else:
            row = self.df_fake.iloc[int(index/2)]
            name1 = row['image_name']
            name2 = row['original']
        image_tensor1 = load_image_as_tensor(os.path.join(self.root_dir, name1), self.image_size)
        image_tensor2 = load_image_as_tensor(os.path.join(self.root_dir, name2), self.image_size)
        return image_tensor1, image_tensor2, class_index

    def __len__(self):
        return min(len(self.df_fake), len(self.df_real)) * 2



from torch.utils.data import DataLoader

def create_data_loaders(root_dir, metadata_df, metadata_real_df, image_size, batch_size, num_workers):
    dataset = VideoDataset(root_dir, metadata_df, metadata_real_df, image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return loader

train_loader = create_data_loaders(ROOT_DIR, metadata_df, metadata_real_df, image_size, batch_size, num_workers=0)



model = Siamese()
model.cuda()



import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(1):
    
    bce_loss = 0.0
    total_examples = 0

    for data in tqdm(train_loader):
        batch_size = data[0].shape[0]
        x1 = data[0].to(gpu)
        x2 = data[1].to(gpu)
        y_true = data[2].to(gpu).float()
        
        optimizer.zero_grad()

        y_pred = model(x1, x2)
        y_pred = y_pred.squeeze()

        loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
        loss.backward()
        optimizer.step()

        batch_bce = loss.item()
        bce_loss += batch_bce * batch_size
        total_examples += batch_size

        print('batch BCE: %.4f' % (batch_bce))

    bce_loss /= total_examples
    print('Epoch: %3d, train BCE: %.4f' % (epoch+1, bce_loss))


print('Finished Training')

torch.save(model.state_dict(), 'binary_classifier_siamese.pth')