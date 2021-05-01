import os
from pathlib import Path
from PIL import Image

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.models as models

from clearml import Task

# ClearML agent
# task = Task.init(project_name='dementia_VR', 
#     task_name='Pseudo eye-tracking data classification with pretrained model')
config_dict = {'num_of_epochs': 10, 'batch_size': 4, 'drop_out': 0.5, 'lr': 2e-5}
# config_dict = task.connect(config_dict)

eyetrack_dir = Path('./data/pseudo_eyetracking')
meta_df_path = os.path.join(eyetrack_dir, 'tain_meta.csv')
classes = ('HC', 'AD')

# dataset class
class EyeTrackDataSet(Dataset):
    def __init__(self, meta_df_path):
        self.file_paths = []
        self.labels = []
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        
        meta_df = pd.read_csv(meta_df_path)
        for _, row in meta_df.iterrows():
            self.file_paths.append(row['path'])
            self.labels.append(row['label'])

    def __getitem__(self, idx):
        eyetrack_img = Image.open(self.file_paths[idx])
        eyetrack_img = eyetrack_img.convert('RGB')
        eyetrack_img = self.transform(eyetrack_img)
        
        return eyetrack_img, self.labels[idx]

    def __len__(self):
        return len(self.file_paths)

# train and test routine
# def train(train_loader, valid_loader, 
#     model, device, optimizer, criterion, 
#     num_epochs, log_interval):
    
#     iters = 0
#     train_loss_insitu = 0.0
#     valid_loss_insitu = 0.0
#     train_loss_list = []
#     valid_loss_list = []

#     model.train()
#     for epoch in range(num_epochs):
#         for batch_idx, (inputs, labels) in enumerate(train_loader):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             # feedforward
#             outputs = model(inputs)

#             # backward & update
#             loss = criterion(outputs, labels)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             # print training stats
#             iters = epoch*len(train_loader) + batch_idx
#             if batch_idx % log_interval == 0:
                

#                 print('Epoch: {} | [{}/{}]'.format(
#                     epoch, iters, num_epochs*len(train_loader)))

# def test(data_loader, model):
#     pass

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
model.fc = nn.Sequential(*[nn.Dropout(p=config_dict.get('drop_out')), nn.Linear(num_features, 2)])

# define optimizer
optimizer = optim.Adam(model.parameters(), lr=config_dict.get('lr'))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config_dict.get('num_of_epochs')//2, gamma=0.5)
criterion = nn.BCELoss()

# device (GPU) setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f'Device to use: {device}')

train_running_loss = 0.0
valid_running_loss = 0.0
train_loss_list = []
valid_loss_list = []

num_epochs = config_dict.get('num_of_epochs')
num_iters = num_epochs*len(train_dl)
log_interval = 100

for epoch in range(config_dict.get('num_of_epochs')):
    # training
    model.train()
    for batch_idx, (inputs, labels) in enumerate(train_dl):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the gradients of parameters    
        optimizer.zero_grad()

        # forward; backward; optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # calculate train loss
        train_running_loss += loss.item()
        if (batch_idx+1) % log_interval == 0:
            avg_train_loss = train_running_loss/log_interval
            train_loss_list.append(avg_train_loss)
            
            print('[{:d}/{:2d}, {:d}/{:d}] Train loss: {:.4f}'.format(
                epoch, num_epochs, batch_idx+1, num_iters, avg_train_loss))
            train_running_loss = 0.0

    # validation    
    print('Start validation')
    model.eval()
    with torch.no_grad():
        for inputs, labels in valid_dl:
            inputs = inputs.to(device)
            labels = inputs.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            valid_running_loss += loss
    
    avg_valid_loss = valid_running_loss/len(valid_dl)
    valid_loss_list.append(avg_valid_loss)
    print('Last train loss: {:.4f}, Valid loss: {:.4f}'.format(
        train_loss_list[-1], avg_valid_loss))

print('Finished training')
