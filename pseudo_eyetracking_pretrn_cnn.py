import os
from pathlib import Path
from PIL import Image

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from clearml import Task

# ClearML agent
# task = Task.init(project_name='dementia_VR', 
#     task_name='Pseudo eye-tracking data classification with pretrained model')
config_dict = {"num_of_epochs": 6, "batch_size": 8, "drop_out": 0.25, "lr": 2e-5}
# config_dict = task.connect(config_dict)

eyetrack_dir = Path('./data/pseudo_eyetracking')
meta_df_path = os.path.join(eyetrack_dir, 'tain_meta.csv')

# dataset class
class EyeTrackDataSet(Dataset):
    def __init__(self, meta_df_path):
        self.file_paths = []
        self.labels = []

        meta_df = pd.read_csv(meta_df_path)
        for _, row in meta_df.iterrows():
            self.file_paths.append(row['path'])
            self.labels.append(row['label'])

    def __getitem__(self, idx):
        eyetrack_img = Image.open(self.file_paths[idx])
        eyetrack_img = eyetrack_img.convert('RGB')
        return eyetrack_img, self.labels[idx]

    def __len__(self):
        return len(self.file_paths)

# train and test routine
def train(train_loader, valid_loader, 
    model, device, optimizer, criterion, 
    num_epochs, log_interval=10):
    
    iters = 0
    train_loss_insitu = 0.0
    valid_loss_insitu = 0.0
    train_loss_list = []
    valid_loss_list = []

    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # feedforward
            outputs = model(inputs)

            # backward & update
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print training stats
            iters = epoch*len(train_loader) + batch_idx
            if batch_idx % log_interval == 0:
                

                print('Epoch: {} | [{}/{}]'.format(
                    epoch, iters, num_epochs*len(train_loader)))

def test(data_loader, model):
    pass

# data sets
train_ds = EyeTrackDataSet(os.path.join(eyetrack_dir, 'train_meta.csv'))
valid_ds = EyeTrackDataSet(os.path.join(eyetrack_dir, 'valid_meta.csv'))
test_ds = EyeTrackDataSet(os.path.join(eyetrack_dir, 'test_meta.csv'))

# data loaders
train_dl = DataLoader(train_ds, batch_size=config_dict.get('batch_size'))
valid_dl = DataLoader(valid_ds, batch_size=config_dict.get('batch_size'))
test_dl = DataLoader(test_ds, batch_size=config_dict.get('batch_size'))

# pretrained model
model = models.resnet18(pretrained=True)

# modify output layer (binary output)
num_features = model.fc.in_features
models.fc = nn.Sequential(*[nn.Dropout(p=config_dict.get('drop_out')), nn.Linear(num_features, 2)])

# define optimizer
optimizer = optim.Adam(model.parameters(), lr=config_dict.get('lr'))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config_dict.get('num_of_epochs')//2, gamma=0.5)
criterion = nn.BCELoss()

# device (GPU) setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f'Device to use: {device}')

for batch_idx, (inputs, labels) in test_dl:
    print(batch_idx)
    print(labels)

    
    