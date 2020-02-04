import os, sys, random
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet

crops_dir = '../deep-faces/faces_224'

gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

metadata_df = pd.read_csv('../deep-faces/metadata.csv')

image_size = 224
batch_size = 64



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



def make_splits(crops_dir, metadata_df, frac):
    real_rows = metadata_df[metadata_df['label'] == 'REAL']
    real_df = real_rows.sample(frac=frac, random_state=666)
    fake_df = metadata_df[metadata_df['original'].isin(real_df['videoname'])]
    
    val_df = pd.concat([real_df, fake_df])
    train_df = metadata_df.loc[~metadata_df.index.isin(val_df.index)]
    return train_df, val_df



from torch.utils.data import DataLoader

def create_data_loaders(crops_dir, metadata_df, image_size, batch_size, num_workers):
    train_df, val_df = make_splits(crops_dir, metadata_df, frac=0.05)

    train_dataset = VideoDataset(crops_dir, train_df, 'train', image_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_dataset = VideoDataset(crops_dir, val_df, 'val', image_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader

train_loader, val_loader = create_data_loaders(crops_dir, metadata_df, image_size, batch_size, num_workers=0)



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



# checkpoint = torch.load('binary_classifier.pth', map_location=gpu)
# model.load_state_dict(checkpoint)

# model.cuda()

# def evaluate(model, data_loader, device):
#     model.train(False)

#     loss = 0
#     total_examples = 0

#     for data in data_loader:
#         with torch.no_grad():
#             batch_size = data[0].shape[0]
#             x = data[0].to(device)
#             y_true = data[1].to(device)
#             y_pred = model(x)
#             loss += criterion(y_pred, y_true)

#         total_examples += batch_size
    
#     loss /= total_examples
#     return loss

# print(evaluate(model, val_loader, gpu))