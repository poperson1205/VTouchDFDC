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

ROOT_DIR = '/media/vtouchinc02/database/RawData/deepfake-faces-retinaface/'

def get_dataframe(root_dir, folder_name):
    crops_dir = os.path.join(root_dir, folder_name)

    with open(os.path.join(crops_dir, 'metadata.json'), 'r') as fp:
        metadata = json.load(fp)

    list_videoname = []
    list_original_width = []
    list_original_height = []
    list_label = []
    list_original = []
    for video_name, attributes in tqdm(metadata.items()):
        if not ('face' in attributes):
            continue

        for frame_index, detection_results in attributes['face'].items():
            for face_index, detection_result in enumerate(detection_results):
                img_name = '%s-%d-%d.png' % (video_name[:-4], int(frame_index), face_index)
                if os.path.isfile(os.path.join(crops_dir, img_name)):
                    list_videoname.append(folder_name + '/' + img_name)
                    list_original_width.append(224)
                    list_original_height.append(224)
                    list_label.append(attributes['label'])
                    if attributes['label'] == 'FAKE':
                        original_video_name = attributes['original']
                        original_img_name = '%s-%d-%d.png' % (original_video_name[:-4], int(frame_index), face_index)
                        list_original.append(folder_name + '/' + original_img_name)
                    else:
                        list_original.append('')

    metadata_df = pd.DataFrame({
        'videoname':list_videoname,
        'original_width':list_original_width,
        'original_height':list_original_height,
        'label':list_label,
        'original':list_original
        })
    
    return metadata_df

# list_metadata_df = []
# for dir_name in tqdm(os.listdir(ROOT_DIR)):
#     print(dir_name)
#     if os.path.isfile(os.path.join(os.path.join(ROOT_DIR, dir_name), 'metadata.json')):
#         list_metadata_df.append(get_dataframe(ROOT_DIR, dir_name))
# metadata_df = pd.concat(list_metadata_df)

# # Save dataframe
# metadata_df.to_csv(os.path.join(ROOT_DIR, 'metadata.csv'))

# Read dataframe
metadata_df = pd.read_csv(os.path.join(ROOT_DIR, 'metadata.csv'))

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
    
    def __init__(self, root_dir, df, split, image_size=224, sample_size=None, seed=None):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size

        if sample_size is not None:
            real_df = df[df['label'] == 'REAL']
            fake_df = df[df['label'] == 'FAKE']
            sample_size = np.min(np.array([sample_size, len(real_df), len(fake_df)]))
            print('%s: sampling %d from %d real videos' % (split, sample_size, len(real_df)))
            print('%s: sampling %d from %d fake videos' % (split, sample_size, len(fake_df)))
            real_df = real_df.sample(sample_size, random_state=seed)
            fake_df = fake_df.sample(sample_size, random_state=seed)
            self.df = pd.concat([real_df, fake_df])
        else:
            self.df = df

        num_real = len(self.df[self.df['label'] == 'REAL'])
        num_fake = len(self.df[self.df['label'] == 'FAKE'])
        print('%s dataset has %d real videos, %d fake videos' % (split, num_real, num_fake))

    def __getitem__(self, index):
        row = self.df.iloc[index]
        file_name = row['videoname']
        class_index = 1 if row['label'] == 'FAKE' else 0
        image_tensor = load_image_as_tensor(os.path.join(self.root_dir, file_name), self.image_size)
        return image_tensor, class_index

    def __len__(self):
        return len(self.df)



def make_splits(metadata_df, frac):
    real_rows = metadata_df[metadata_df['label'] == 'REAL']
    real_df = real_rows.sample(frac=frac, random_state=666)
    fake_df = metadata_df[metadata_df['original'].isin(real_df['videoname'])]
    
    val_df = pd.concat([real_df, fake_df])
    train_df = metadata_df.loc[~metadata_df.index.isin(val_df.index)]
    return train_df, val_df



from torch.utils.data import DataLoader

def create_data_loaders(root_dir, metadata_df, image_size, batch_size, num_workers):
    train_df, val_df = make_splits(metadata_df, frac=0.001)

    train_dataset = VideoDataset(root_dir, train_df, 'train', image_size, sample_size=10000)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_dataset = VideoDataset(root_dir, val_df, 'val', image_size, sample_size=100, seed=1234)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader

train_loader, val_loader = create_data_loaders(ROOT_DIR, metadata_df, image_size, batch_size, num_workers=0)



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
#             y_true = data[1].to(device).float()
#             y_pred = model(x).squeeze()
#             loss += F.binary_cross_entropy_with_logits(y_pred, y_true).item() * batch_size

#         total_examples += batch_size
    
#     loss /= total_examples
#     return loss

# print(evaluate(model, val_loader, gpu))