import cv2
from pathlib2 import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as transforms
import torchvision.models as models

from clearml import Task

##### connect ClearML agent -----
task = Task.init(project_name="VR Mental Health Clinic", 
                 task_name="Audio classification using recurrence plot")
config_dict = {'num_of_epochs': 10, 'batch_size': 4, 'drop_out': 0.25, 'lr': 2e-5,
               'save_path': Path('./results/20210520-auio-rp-cnn')}
config_dict = task.connect(config_dict)

##### data preparation -----
class AudioRpDataset(Dataset):
    def __init__(self, meta_csv_path):
        self.file_paths = []
        self.labels = []
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

        meta_df = pd.read_csv(meta_csv_path)
        for _, row in meta_df.iterrows():
            self.file_paths.append(row['path'])
            self.labels.append(row['label'])

    def __getitem__(self, idx):
        rp_img = cv2.imread(self.file_paths[idx])
        rp_img = cv2.cvtColor(rp_img, cv2.COLOR_BGR2RGB)
        rp_img = self.transform(rp_img)

        return rp_img, self.labels[idx]

    def __len__(self):
        return len(self.file_paths)

audio_rp_dir = Path('./data/audio')
train_ds = AudioRpDataset(str(audio_rp_dir/'meta_train.csv'))
valid_ds = AudioRpDataset(str(audio_rp_dir/'meta_valid.csv'))
test_ds = AudioRpDataset(str(audio_rp_dir/'meta_test.csv'))

train_dl = DataLoader(train_ds, batch_size=config_dict.get('batch_size'))
valid_dl = DataLoader(valid_ds, batch_size=config_dict.get('batch_size'))
test_dl = DataLoader(test_ds, batch_size=config_dict.get('batch_size'))

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
tensorboard_writer = SummaryWriter("./tensorboard_logs")

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
            total += labels.size(0)
            outputs = (outputs > 0.5).float()
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
fig, ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax, cmap='Blues')

ax.set_title("Confusion matrix")
ax.set_xlabel("Predicted labels")
ax.set_ylabel("True labels")
ax.xaxis.set_ticklabels(['AD', 'HC'])
ax.yaxis.set_ticklabels(['AD', 'HC'])

##### save results -----
save_path = config_dict.get('save_path')
if save_path == None:
    pass
else:
    torch.save(model.state_dict(), str(save_path/'model.pt'))
    fig.savefig(str(save_path/'confusion_matrix.png'))
