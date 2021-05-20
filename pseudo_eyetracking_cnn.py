import os
from pathlib2 import Path
from PIL import Image

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from clearml import Task

##### ClearML agent -----
task = Task.init(project_name='VR Mental Health Clinic', 
    task_name='Pseudo eye-tracking data classification with pretrained model')
config_dict = {'num_of_epochs': 10, 'batch_size': 8, 'drop_out': 0.25, 'lr': 5e-5,
               'save_dir': Path('./results/20210520-peudo-eyetracking-cnn')}
config_dict = task.connect(config_dict)
print(config_dict)

eyetrack_dir = Path('./data/pseudo_eyetracking')

##### dataset class -----
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
    
##### data preparation -----
def custom_imshow(img):
    img = (img/2) + 0.5 # unnormalize
    npimg = img.numpy() 
    plt.imshow(np.transpose(npimg, (1,2,0))) # reshape into (h,w,c)
    plt.show()

# data sets
train_ds = EyeTrackDataSet(os.path.join(eyetrack_dir, 'train_meta.csv'))
valid_ds = EyeTrackDataSet(os.path.join(eyetrack_dir, 'valid_meta.csv'))
test_ds = EyeTrackDataSet(os.path.join(eyetrack_dir, 'test_meta.csv'))

# data loaders
train_dl = DataLoader(train_ds, batch_size=config_dict.get('batch_size'))
valid_dl = DataLoader(valid_ds, batch_size=config_dict.get('batch_size'))
test_dl = DataLoader(test_ds, batch_size=config_dict.get('batch_size'))
    
# dataiter = iter(train_dl)
# images, labels = dataiter.next()
# custom_imshow(torchvision.utils.make_grid(images))

##### model preparation -----
# pretrained model
model = models.resnet18(pretrained=True)
model.fc = nn.Sequential(nn.Dropout(p=config_dict.get('drop_out')),
                         nn.Linear(512,1),
                         nn.Sigmoid())

# define optimizer
optimizer = optim.Adam(model.parameters(), lr=config_dict.get('lr'))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config_dict.get('num_of_epochs')//4, gamma=0.5)
criterion = nn.BCELoss()

# device (GPU) setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f'Device to use: {device}')

##### training -----
train_running_loss = 0.0
valid_running_loss = 0.0
valid_acc = 0.0
train_loss_list = []
valid_loss_list = []
valid_acc_list = []

num_epochs = config_dict.get('num_of_epochs')
log_interval = 50
tensorboard_writer = SummaryWriter('./tensorboard_logs')

for epoch in range(config_dict.get('num_of_epochs')):
    # training
    model.train()
    for batch_idx, (inputs, labels) in enumerate(train_dl):        
        inputs = inputs.to(device)
        labels = labels.type(torch.FloatTensor)
        labels = labels.to(device)

        # zero the gradients of parameters    
        optimizer.zero_grad()

        # forward; backward; optimize
        outputs = model(inputs)
        loss = criterion(torch.squeeze(outputs, 1), labels)
        loss.backward()
        optimizer.step()

        # calculate train loss
        train_running_loss += loss.item()
        if (batch_idx+1) % log_interval == 0:
            avg_train_loss = train_running_loss/log_interval
            train_loss_list.append(avg_train_loss)
            
            print('[{:d}/{:2d}, {:d}/{:d}] Train loss: {:.4f}'.format(
                epoch+1, num_epochs, batch_idx+1, len(train_dl), avg_train_loss))
            
            # tensor board
            iteration = epoch*len(train_dl) + (batch_idx+1)
            tensorboard_writer.add_scalar('Training loss', avg_train_loss, iteration)
            tensorboard_writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], iteration)

            train_running_loss = 0.0

    # validation for each epoch
    print('Start validation')
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(valid_dl):
            inputs = inputs.to(device)
            labels = labels.type(torch.FloatTensor)
            labels = labels.to(device)
            
            # forward
            outputs = model(inputs)
            outputs = torch.squeeze(outputs, 1)
            
            # loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss
            
            # prediction
            outputs = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (outputs == labels).sum().item()
            
    avg_valid_loss = valid_running_loss/len(valid_dl)
    valid_loss_list.append(avg_valid_loss)
    valid_acc = correct/total
    valid_acc_list.append(valid_acc)
    print('Last train loss: {:.4f}, Valid loss: {:.4f}, Accuracy: {:.2%}'.format(
        train_loss_list[-1], avg_valid_loss, valid_acc))
    
    # tensor board
    tensorboard_writer.add_scalar('Valid loss', avg_valid_loss, epoch)
    tensorboard_writer.add_scalar('Accuracy', valid_acc, epoch)

    train_running_loss = 0.0
    valid_running_loss = 0.0
    valid_acc = 0.0
    scheduler.step()

print('Finished training')

##### testing -----
print('Start testing')
y_true = []
y_pred = []
model.eval()
with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(test_dl):
        inputs = inputs.to(device)
        labels = labels.type(torch.FloatTensor)
        labels = labels.to(device)
        
        # forward
        outputs = model(inputs)
        outputs = torch.squeeze(outputs, 1)        
        outputs = (outputs > 0.5).float()

        y_true.extend(labels.tolist())
        y_pred.extend(outputs.tolist())

print('Classification report')
print(metrics.classification_report(y_true, y_pred, labels=[1,0], digits=4))

cm = metrics.confusion_matrix(y_true, y_pred, labels=[1,0])
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, ax=ax, cmap='Blues')

ax.set_title("Confusion matrix")
ax.set_xlabel("Predicted labels")
ax.set_ylabel("True labels")
ax.xaxis.set_ticklabels(['AD', 'HC'])
ax.yaxis.set_ticklabels(['AD', 'HC'])

##### save results -----
save_dir = config_dict.get('save_dir')
if save_dir == None:
    pass
else:
    torch.save(model.state_dict(), str(save_dir/'model3.pt'))
    fig.savefig(str(save_dir/'confusion_matrix3.png'))
