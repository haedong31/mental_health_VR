from PIL import Image
from pathlib import Path
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from clearml import Task

task = Task.init(project_name='VR Mental Health Clinic',
                 task_name='Audio IFS Plots - Pretrained ResNet')
config_dict = {'num_of_epochs': 10, 'batch_size': 4, 'drop_out': 0.25, 'lr': 5e-5,
               'save_dir': Path('./results')}
config_dict = task.connect(config_dict)
print(config_dict)

data_dir = Path('./data/audio/ifs')

class AudioIFSDataSet(Dataset):
    def __init__(self,meta_df_path,data_dir,h,w):
        self.paths = []
        self.labels = []
        self.transform = transforms.Compose(
            [transforms.Resize((h,w)),
             transforms.ToTensor(),
             transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        
        meta_df = pd.read_csv(meta_df_path)
        num_id = 8
        for _, row in meta_df.iterrows():
            f = row['file']
            l = row['label']
            self.paths.append(data_dir/(f+'.png'))
            self.labels.append(l)
            
            for i in range(num_id):
                self.paths.append(data_dir/(f+'_'+str(i+1)+'.png'))
                self.labels.append(l)
                
                for j in range(num_id):
                    self.paths.append(data_dir/(f+'_'+str(i+1)+str(j+1)+'.png'))
                    self.labels.append(l)
                    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self,idx):
        ifs_img = Image.open(self.paths[idx])
        ifs_img = ifs_img.convert('RGB')
        ifs_img = self.transform(ifs_img)
        
        return ifs_img, self.labels[idx]
    
# data sets
h = 1200
w = 1260
train_ds = AudioIFSDataSet('data/audio/meta_train.csv', data_dir, h, w) 
valid_ds = AudioIFSDataSet('data/audio/meta_valid.csv', data_dir, h, w)
test_ds = AudioIFSDataSet('data/audio/meta_test.csv', data_dir, h, w)

# data loaders
train_dl = DataLoader(train_ds, batch_size=config_dict.get('batch_size'))
valid_dl = DataLoader(valid_ds, batch_size=config_dict.get('batch_size'))
test_dl = DataLoader(test_ds, batch_size=config_dict.get('batch_size'))

# pretrained ResNet
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) # try from the empty model? put this option in the configuratin dictionary?
model.fc = nn.Sequential(nn.Dropout(p=config_dict.get('drop_out')),
                         nn.Linear(512,1),
                         nn.Sigmoid())

# optimizer
optimizer = optim.Adam(model.parameters(), lr=config_dict.get('lr'))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config_dict.get('num_of_epochs')//4, gamma=0.5)
criterion = nn.BCELoss()

# device (GPU) setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f'Device to use: {device}')

train_running_loss = 0.0
valid_running_loss = 0.0
valid_running_acc = 0.0
train_loss = []
valid_loss = []
valid_acc = []

num_epochs = config_dict.get('num_of_epochs')
log_interval = 50
tb_writer = SummaryWriter('./tensorboard_logs')

for epoch in range(num_epochs):
    ### training loop
    model.train()
    for batch_idx, (inputs,labels) in enumerate(train_dl):
        inputs = inputs.to(device)
        labels = labels.type(torch.FloatTensor)
        labels = labels.to(device)

        # zeroing the gradients of parameters
        optimizer.zero_grad()

        # train routine
        outputs = model(inputs) # forward
        loss = criterion(torch.squeeze(outputs,1), labels)
        loss.backward() # backward
        optimizer.step() # optimize

        # calculate train loss
        train_running_loss += loss.item()
        if (batch_idx+1) % log_interval == 0:
            avg_train_loss = train_running_loss/log_interval
            train_loss.append(avg_train_loss)

            print('[{:d}/{:2d}, {:d}/{:d}] Train loss: {:.4f}'.format(
                epoch+1, num_epochs, batch_idx+1, len(train_dl), avg_train_loss))

            # write in the tensorboard
            iteration = epoch*len(train_dl) + (batch_idx+1)
            tb_writer.add_scalar('Training loss', avg_train_loss, iteration)
            tb_writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], iteration)

            train_running_loss = 0.0

    ### validation
    print('Start Validation')
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
    valid_loss.append(avg_valid_loss)
    valid_running_acc = correct/total
    valid_acc.append(valid_running_acc)
    print('Last train loss: {:.4f}, Valid loss: {:.4f}, Accuracy: {:.2%}'.format(
        train_loss[-1], avg_valid_loss, valid_running_acc))
    
    # tensor board
    tb_writer.add_scalar('Valid loss', avg_valid_loss, epoch)
    tb_writer.add_scalar('Accuracy', valid_running_acc, epoch)

    train_running_loss = 0.0
    valid_running_loss = 0.0
    valid_running_acc = 0.0
    scheduler.step()

### testing
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
