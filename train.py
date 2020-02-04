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



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.avg_pool = nn.AvgPool2d(53)
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.avg_pool(self.pool(F.relu(self.conv2(x))))
        x = x.view(-1, 16)
        x = F.relu(self.fc(x))
        return x

net = Net()



import torch.optim as optim

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# for epoch in range(2):
    
#     running_loss = 0.0
#     for i, data in enumerate(train_loader, 0):
#         inputs, labels = data
        
#         optimizer.zero_grad()

#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         if i % 1421 == 1420:
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0

# print('Finished Training')

# torch.save(net.state_dict(), 'binary_classifier.pth')



checkpoint = torch.load('binary_classifier.pth', map_location=gpu)
net.load_state_dict(checkpoint)

net.cuda()

def evaluate(net, data_loader, device):
    net.train(False)

    loss = 0
    total_examples = 0

    for data in data_loader:
        with torch.no_grad():
            batch_size = data[0].shape[0]
            x = data[0].to(device)
            y_true = data[1].to(device)
            y_pred = net(x)
            loss += criterion(y_pred, y_true)

        total_examples += batch_size
    
    loss /= total_examples
    return loss

print(evaluate(net, val_loader, gpu))