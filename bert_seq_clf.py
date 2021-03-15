import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from transformers import BertTokenizer, BertForSequenceClassification
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
import torch
import torch.nn as nn
import torch.optim as optim

## Prepare data -----
# BERT for text embedding
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_seq_len = 128
batch_size = 4
pad_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
unk_idx = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

source_folder = os.getcwd()
destination_folder = os.getcwd()

# define fields
label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False,
    batch_first=True, fix_length=max_seq_len, pad_token=pad_idx, unk_token=unk_idx)
fields = [("text", text_field), ("label", label_field)]

# create tabular dataset
train, valid, test = TabularDataset.splits(path=source_folder, train="train.csv", validation="valid.csv",
    test="test.csv", format="CSV", fields=fields, skip_header=True)

# create iterators
train_iter = BucketIterator(train, batch_size=batch_size, sort_key=lambda x: len(x.text),
    device=device, train=True, sort=True, sort_within_batch=True)
valid_iter = BucketIterator(valid, batch_size=batch_size, sort_key=lambda x: len(x.text),
    device=device, train=True, sort=True, sort_within_batch=True)
test_iter = Iterator(test, batch_size=batch_size, device=device, train=False, shuffle=False, sort=False)

## Build model -----
class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BertForSequenceClassification.from_pretrained("bert-base-cased")

    def forward(self, text, label):
        loss, output = self.encoder(text, labels=label)[:2]

        return loss, output

## Training and evalutation functions -----
def save_checkpoint(save_path, model, valid_loss):
    if save_path == None:
        return

    state_dict = {"model_state_dict": model.state_dict(), "valid_loss": valid_loss}

    torch.save(state_dict, save_path)
    print(f"Model saved to ==> {save_path}")

def load_checkpoint(load_path, model):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f"Model loaded from <== {load_path}")

    model.load_state_dict(state_dict["model_state_dict"])
    return state_dict["valid_loss"]

def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path == None:
        return

    state_dict = {"train_loss_list": train_loss_list,
                  "valid_loss_list": valid_loss_list,
                  "global_steps_list": global_steps_list}

    torch.save(state_dict, save_path)
    print(f"Model saved to ==> {save_path}")

def load_metrics(load_path):
    if load_path == None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f"Model loaded from <== {load_path}")

    train_loss_list = state_dict["train_loss_list"]
    valid_loss_list = state_dict["valid_loss_list"]
    global_steps_list = state_dict["global_steps_list"]

    return train_loss_list, valid_loss_list, global_steps_list

def train_model(model, optimizer, file_path=destination_folder,
                train_loader=train_iter, valid_loader=valid_iter, num_epochs=5,
                eval_every=len(train_iter) // 2, best_valid_loss=float("Inf")):
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for (text, label), _ in train_loader:
            label = label.type(torch.LongTensor)
            label = label.to(device)
            text = text.type(torch.LongTensor)
            text = text.to(device)

            # feedforward            
            output = model(text, label)
            loss, _ = output

            # update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # validation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    # validation loop
                    for (text, label), _ in valid_loader:
                        label = label.type(torch.LongTensor)
                        label = label.to(device)
                        text = text.type(torch.LongTensor)
                        text = text.to(device)
                        
                        output = model(text, label)
                        loss, _ = output

                        valid_running_loss += loss.item()
                
                # validation
                avg_train_loss = running_loss / eval_every
                avg_valid_loss = valid_running_loss / len(valid_loader)

                train_loss_list.append(avg_train_loss)
                valid_loss_list.append(avg_valid_loss)
                global_steps_list.append(global_step)

                # reset running values and model
                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()

                # print progress
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{global_step}/{num_epochs*len(train_loader)}] \
                    Train loss: {avg_train_loss:.4f}, Valid loss: {avg_valid_loss:.4f}")

                # check point
                if best_valid_loss > avg_valid_loss:
                    best_valid_loss = avg_valid_loss
                    save_checkpoint(os.path.join(file_path,"model.pt"), model, best_valid_loss)
                    save_metrics(os.path.join(file_path,"metrics.pt"), train_loss_list, valid_loss_list, global_steps_list)

    save_metrics(os.path.join(file_path,"metrics.pt"), train_loss_list, valid_loss_list, global_steps_list)
    print("Training finished")

def evaluate(model, test_loader=test_iter):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (text, label), _ in test_loader:
            label = label.type(torch.LongTensor)
            label = label.to(device)
            text = text.type(torch.LongTensor)
            text = text.to(device)
            
            output = model(text, label)
            loss, output = output
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_true.extend(label.tolist())

    print("Classification report:")
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap="Blues", fmt="d")

    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.xaxis.set_ticklabels(["FAKE", "TRUE"])
    ax.yaxis.set_ticklabels(["FAKE", "TRUE"])

## Run -----
model = BertClassifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)

train_model(model, optimizer)

# visualization
train_loss_list, valid_loss_list, global_steps_list = load_metrics(os.path.join(destination_folder, "metrics.pt"))

plt.plot(global_steps_list, train_loss_list, label="Train")
plt.plot(global_steps_list, valid_loss_list, label="Valid")
plt.xlabel("Global steps")
plt.ylabel("Loss")
plt.legend()
plt.show()

best_model = BertClassifier().to(device)
load_checkpoint(os.path.join(destination_folder, "model.pt"), best_model)
evaluate(best_model, test_iter)
