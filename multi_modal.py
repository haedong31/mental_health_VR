from PIL import Image
from PIL import ImageOps
from pathlib2 import Path

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

import torchvision.models as models
import torchvision.transforms as transforms

from clearml import Task

##### ClearML agent -----
config_dict = {'k_folds': 10, 'num_of_epochs': 8, 'batch_size': 8,
    'lr1': 2e-5, 'lr2': 2e-5, 'drop_out': 0.25, 'save_dir': './results/20210523-multi-10cv'}
task = Task.init(project_name='VR Mental Health Clinic', 
    task_name='Multi-modal nonlinguistic classification')
config_dict = task.connect(config_dict)
print(config_dict)

##### data preparation -----
class MultiModalDataset(Dataset):
    def __init__(self, meta_csv_path):
        super().__init__()
        self.eye_paths = []
        self.rp_paths = []
        self.labels = []
        self.num_row_meta_df = 0
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

        meta_df = pd.read_csv(meta_csv_path)
        self.num_row_meta_df = meta_df.shape[0]
        for _, row in meta_df.iterrows():
            self.eye_paths.append(Path(row['eye_path']))
            self.rp_paths.append(Path(row['rp_path']))
            self.labels.append(row['label'])

    def crop_box(self, img):
        inverted = ImageOps.invert(img)
        img_box = inverted.getbbox()
        cropped = img.crop(img_box)
        
        return cropped

    def __getitem__(self, index):
        eye_img = Image.open(str(self.eye_paths[index])).convert('RGB')
        eye_img = self.crop_box(eye_img)
        eye_img = self.transform(eye_img)

        rp_img = Image.open(str(self.rp_paths[index])).convert('RGB')
        rp_img = self.crop_box(rp_img)
        rp_img = self.transform(rp_img)
        
        return eye_img, rp_img, self.labels[index]
        
    def __len__(self):
        return self.num_row_meta_df

##### data preparation -----
cwd = Path.cwd()
train_ds = MultiModalDataset(str(cwd/'data'/'meta_train.csv'))
test_ds = MultiModalDataset(str(cwd/'data'/'meta_test.csv'))

kfold = KFold(n_splits=config_dict.get('k_folds'), shuffle=True)
num_epochs = config_dict.get('num_of_epochs')

###### training -----
# device (GPU) setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device to use: {device}')

# tensroboard writer
tensorboard_writer = SummaryWriter('./tensorboard_logs')

eye_precisions = []
eye_recalls = []
eye_f1_scores = []
eye_accuracies = []

rp_precisions = []
rp_recalls = []
rp_f1_scores = []
rp_accuracies = []

save_dir = Path(config_dict.get('save_dir'))
save_dir.mkdir(parents=True, exist_ok=True)

for k, (train_idx, valid_idx) in enumerate(kfold.split(train_ds)):
    print(f'{k+1}-fold')

    # prepare kth fold data
    train_subsampler = SubsetRandomSampler(train_idx)
    valid_subsampler = SubsetRandomSampler(valid_idx)
    
    train_dl = DataLoader(train_ds, batch_size=config_dict.get('batch_size'), sampler=train_subsampler)
    valid_dl = DataLoader(train_ds, batch_size=config_dict.get('batch_size'), sampler=valid_subsampler)

    log_interval = int(np.floor(len(train_dl)/3))

    # define models
    eye_model = models.resnet18(pretrained=True)
    eye_model.fc = nn.Sequential(nn.Dropout(p=config_dict.get('drop_out')),
                                 nn.Linear(512,1),
                                 nn.Sigmoid())
    eye_model.to(device)

    rp_model = models.resnet18(pretrained=True)
    rp_model.fc = nn.Sequential(nn.Dropout(p=config_dict.get('drop_out')),
                                nn.Linear(512,1),
                                nn.Sigmoid())
    rp_model.to(device)

    # define loss functions
    eye_criterion = nn.BCELoss()
    rp_criterion = nn.BCELoss()

    # define optimizers
    eye_optimizer = optim.Adam(eye_model.parameters(), lr=config_dict.get('lr1'))
    rp_optimizer = optim.Adam(rp_model.parameters(), lr=config_dict.get('lr2'))

    # define learning rate scheduler
    scheduler1 = optim.lr_scheduler.StepLR(eye_optimizer, step_size=num_epochs//4, gamma=0.5)
    scheduler2 = optim.lr_scheduler.StepLR(rp_optimizer, step_size=num_epochs//4, gamma=0.5)
    
    iterations = 0
    eye_train_running_loss = 0.0
    rp_train_running_loss = 0.0
    for epoch in range(num_epochs):
        eye_model.train()
        rp_model.train()

        for batch_idx, (eye_inputs, rp_inputs, labels) in enumerate(train_dl):
            # put data into device
            eye_inputs = eye_inputs.to(device)
            rp_inputs = rp_inputs.to(device)
            labels = labels.type(torch.FloatTensor)
            labels = labels.to(device)

            # forward
            eye_outputs = eye_model(eye_inputs)
            rp_outputs = rp_model(rp_inputs)

            # backward and optimization
            eye_loss = eye_criterion(torch.squeeze(eye_outputs,1), labels)
            eye_loss.backward()
            eye_optimizer.step()
            eye_optimizer.zero_grad()

            rp_loss = rp_criterion(torch.squeeze(rp_outputs,1), labels)
            rp_loss.backward()
            rp_optimizer.step()
            rp_optimizer.zero_grad()
            
            eye_train_running_loss += eye_loss.item()
            rp_train_running_loss += rp_loss.item()

            if (batch_idx+1) % log_interval == 0:
                iterations += 1
                eye_avg_loss = eye_train_running_loss/log_interval
                rp_avg_loss = rp_train_running_loss/log_interval

                print('[{:d}/{:2d}, {:d}/{:d}] Eye train loss: {:.4f} RP train loss: {:.4f}'.format(
                epoch+1, num_epochs, batch_idx+1, len(train_dl), eye_avg_loss, rp_avg_loss))
                
                tensorboard_writer.add_scalars('Train loss', {
                    'Eye': eye_avg_loss,
                    'RP': rp_avg_loss},
                    iterations)
                
                eye_train_running_loss = 0.0
                rp_train_running_loss = 0.0

    print('Start evaluation')               
    y_true = []
    eye_pred = []
    rp_pred = []

    eye_model.eval()
    rp_model.eval()

    with torch.no_grad():
        for batch_idx, (eye_inputs, rp_inputs, labels) in enumerate(valid_dl):
            # put data into device
            eye_inputs = eye_inputs.to(device)
            rp_inputs = rp_inputs.to(device)
            labels = labels.type(torch.FloatTensor)
            labels = labels.to(device)
            
            y_true.extend(labels.tolist())
            
            # forward
            eye_outputs = eye_model(eye_inputs)
            eye_outputs = torch.squeeze(eye_outputs, 1)
            eye_outputs = (eye_outputs > 0.5).float()
            eye_pred.extend(eye_outputs.tolist())

            rp_outputs = rp_model(rp_inputs)
            rp_outputs = torch.squeeze(rp_outputs, 1)
            rp_outputs = (rp_outputs > 0.5).float()
            rp_pred.extend(rp_outputs.tolist())

    print('Eye')
    print(metrics.classification_report(y_true, eye_pred, labels=[1,0], digits=4))
    print('RP')
    print(metrics.classification_report(y_true, rp_pred, labels=[1,0], digits=4))

    eye_precisions.append(metrics.precision_score(y_true, eye_pred))
    eye_recalls.append(metrics.recall_score(y_true, eye_pred))
    eye_f1_scores.append(metrics.f1_score(y_true, eye_pred))
    eye_acc = metrics.accuracy_score(y_true, eye_pred)
    eye_accuracies.append(eye_acc)

    rp_precisions.append(metrics.precision_score(y_true, rp_pred))
    rp_recalls.append(metrics.recall_score(y_true, rp_pred))
    rp_f1_scores.append(metrics.f1_score(y_true, rp_pred))
    rp_acc = metrics.accuracy_score(y_true, rp_pred)
    rp_accuracies.append(rp_acc)
    
    tensorboard_writer.add_scalars('Accuracy',{
        'Eye': eye_acc,
        'RP': rp_acc},
        epoch+1)
    
    if save_dir == None:
        pass
    else:
        torch.save(eye_model.state_dict(), str(save_dir/f'eye_model_{k}'))
        torch.save(rp_model.state_dict(), str(save_dir/f'rp_model_{k}'))
