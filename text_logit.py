import pandas as pd
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

def trunc_pad(s, trunc_len):
    # max length of sentences
    max_len = 0
    for v in s.values:
        if len(v) > max_len:
            max_len = len(v)
    
    # truncate sentences longer than `trunc_len`
    if max_len > trunc_len:
        long_sen_idx = list()
        for n, v in enumerate(s.values):
            if len(v) > trunc_len:
                long_sen_idx.append(n)
        
        s = s.drop(s.index[long_sen_idx])
        
        # renew max length of sentences
        max_len = 0
        for v in s.values:
            if len(v) > max_len:
                max_len = len(v)
    else:
        long_sen_idx = None
    
    # zero padding
    padded = np.array([v + [0]*(max_len - len(v)) for v in s.values])
    
    return padded, long_sen_idx
    
# import dataset
df = pd.read_csv("./data/cookie_ebd_prep.csv", index_col=0)

# load model and tokenizer
model = BertModel.from_pretrained("bert-base-cased")
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
max_input_size = 512

# tokenization {101: [CLS], 102: [SEP]}
tokenized = df["utts"].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

# truncate too long sentences and zero padding
padded, long_sen_idx = trunc_pad(tokenized, max_input_size)

# attention mask
att_mask = np.where(padded == 0, 0, 1)

# embedding
padded = torch.tensor(padded).to(torch.int64)
att_mask = torch.tensor(att_mask)
with torch.no_grad():
    ebdded = model(padded, attention_mask = att_mask)

# only use embedding for [CLS] token
cls_features = ebdded[0][:,0,:].numpy()

# split training test datasets
labels = df["labels"]
labels = labels.drop(labels.index[long_sen_idx])
labels = np.array(labels)

# training and prediction of logistic regression
num_iters = 30
precisions = list()
recalls = list()
f1_scores = list()
accuracies = list()
for i in range(num_iters):
    trnx, testx, trny, testy = train_test_split(cls_features, labels)
    
    # train logistic regression
    lr_clf = LogisticRegression(max_iter=10000)
    lr_clf.fit(trnx, trny)
    
    # prediction
    predy = lr_clf.predict(testx)
    
    # performance measure
    precisions.append(metrics.precision_score(testy, predy))
    recalls.append(metrics.recall_score(testy, predy))
    f1_scores.append(metrics.f1_score(testy, predy))
    accuracies.append(metrics.accuracy_score(testy, predy))

# save results
result_df = pd.DataFrame(data={"precision": precisions, "recall": recalls,
                               "f1": f1_scores, "accuracy": accuracies})
result_df.to_excel("./logistic.xlsx")
boxplot = result_df.boxplot()
