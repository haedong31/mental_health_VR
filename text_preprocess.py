from transformers import BertTokenizer, BertModel
from bert_vocab_check import BertVocabCheck

from glob import glob
from pathlib import Path
import re
import os

from sklearn.model_selection import train_test_split
from expand_contraction import expand_contraction
import pandas as pd
import numpy as np

## custom functions -----
def read_and_concat(source_dir, group):
    if group == "control":
        paths = glob(os.path.join(source_dir, "control", "*.txt"))
    elif group == "experimental":
        paths = glob(os.path.join(source_dir, "dementia", "*.txt"))
    else:
        raise ValueError("Wrong group name")
    
    concat_utts = list()
    for path in paths:
        with open(path) as f:
            utt = f.read().splitlines()
            concat_utts.append(" ".join(utt))
    
    return concat_utts, paths

def save_oov(oov_dict, save_name):
    k = list(oov_dict.keys())
    v = list(oov_dict.values()) 
    oov_df = pd.DataFrame(data={"word": k, "count": v})

    save_path = "./results/oov/" + save_name + ".xlsx"
    oov_df.to_excel(save_path)

def pattern_sub(pattern, sub, ds):
    pattern_compile = re.compile(pattern)
    for n, s in enumerate(ds):
        ds[n] = pattern_compile.sub(sub, s)

    return ds

def comp_word_expand(con, ds):
    p = "[a-z]*\\"+con+"[a-z]*"
    pattern_compile = re.compile(p)
    for n, s in enumerate(ds):
        matched_words = pattern_compile.findall(s)

        for mword in matched_words:
            expand_word = mword.replace(con, " ")
            s = s.replace(mword, expand_word)
        ds[n] = s
    
    return ds

## import data -----
source_dir = Path("./data/text/cookie_minimal_prep")
con_utts, con_paths = read_and_concat(source_dir, "control")
exp_utts, exp_paths = read_and_concat(source_dir, "experimental")

# check some words in BERT vocab
tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
# 'she' in tokenizer.vocab # True
# 'I' in tokenizer.vocab # False
# 'he' in tokenizer.vocab # True
# 'is' in tokenizer.vocab # True
# 'are' in tokenizer.vocab # True
# 'overflow' in tokenizer.vocab # False
# '.' in tokenizer.vocab # True
# '?' in tokenizer.vocab # True
'flowing' in tokenizer.vocab

## 1st run -----
print("\n ----- 1st run ----- \n")

print("control group")
con_vocab = BertVocabCheck(con_utts)
print("experimental group")
exp_vocab = BertVocabCheck(exp_utts)

con_oov = con_vocab.oov
exp_oov = exp_vocab.oov

# top ten OOV words
print(list(con_oov.items())[:10])
print(list(exp_oov.items())[:10])

# save OOV
# save_oov(con_oov, "con1")
# save_oov(exp_oov, "exp1")

# exapand contractions
con_utts = expand_contraction(con_utts)
exp_utts = expand_contraction(exp_utts)

# remove 's from possesive constractions
con_utts = pattern_sub("\'s", "", con_utts)
exp_utts = pattern_sub("\'s", "", exp_utts)

## 2nd run -----
print("\n ----- 2nd run ----- \n")

print("control group")
con_vocab = BertVocabCheck(con_utts)
print("experimental group")
exp_vocab = BertVocabCheck(exp_utts)

con_oov = con_vocab.oov
exp_oov = exp_vocab.oov

print(list(con_oov.items())[:10])
print(list(exp_oov.items())[:10])

# save_oov(con_oov, "con2")
# save_oov(exp_oov, "exp2")

# expand words concatednated by _
con_utts = comp_word_expand("_", con_utts)
exp_utts = comp_word_expand("_", exp_utts)

con_utts = comp_word_expand("+", con_utts)
exp_utts = comp_word_expand("+", exp_utts)

# substitute some patterns
con_utts = pattern_sub("splashing", "splash", con_utts)
exp_utts = pattern_sub("splashing", "splash", exp_utts)

con_utts = pattern_sub("stools", "stool", con_utts)
exp_utts = pattern_sub("stools", "stool", exp_utts)

con_utts = pattern_sub("ladys", "lady", con_utts)
exp_utts = pattern_sub("ladys", "lady", exp_utts)

con_utts = pattern_sub("knobs", "knob", con_utts)
exp_utts = pattern_sub("knobs", "knob", exp_utts)

con_utts = pattern_sub("cupboards", "cupboard", con_utts)
exp_utts = pattern_sub("cupboards", "cupboard", exp_utts)

con_utts = pattern_sub("jars", "jar", con_utts)
exp_utts = pattern_sub("jars", "jar", exp_utts)

con_utts = pattern_sub(":", "", con_utts)
exp_utts = pattern_sub(":", "", exp_utts)

## 3rd run -----
print("\n ----- 3rd run ----- \n")

print("control group")
con_vocab = BertVocabCheck(con_utts)
print("experimental group")
exp_vocab = BertVocabCheck(exp_utts)

con_oov = con_vocab.oov
exp_oov = exp_vocab.oov

print(list(con_oov.items())[:10])
print(list(exp_oov.items())[:10])

# save_oov(con_oov, "con3")
# save_oov(exp_oov, "exp3")

con_utts = pattern_sub("overflowing", "over flow", con_utts)
exp_utts = pattern_sub("overflowing", "over flow", exp_utts)

# conf = "./data/cookie_ebd_prep_con.txt"
# with open(conf, "wt") as f:
#     for u in con_utts:
#         f.write(u+"\n")

# expf = "./data/cookie_ebd_prep_exp.txt"
# with open(expf, "wt") as f:
#     for u in exp_utts:
#         f.write(u+"\n")

utts = con_utts + exp_utts
labels = [0]*len(con_utts) + [1]*len(exp_utts)
paths = con_paths + exp_paths


df = pd.DataFrame(data={"utts": utts, "labels": labels, "paths": paths})
df = df.sample(frac = 1)
df.to_csv("./data/text/cookie_ebd_prep.csv", index=False)

## train-test split -----
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
df = pd.read_csv("./data/text/cookie_ebd_prep.csv", index_col=0)
df["labels"].value_counts()

# load model and tokenizer
model = BertModel.from_pretrained("bert-base-cased")
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
max_input_size = 512

 # tokenization {101: [CLS], 102: [SEP]}
tokenized = df["utts"].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

# sequence lengh distribution
seq_lens = []
for s in tokenized:
    seq_lens.append(len(s))
seq_lens = pd.Series(seq_lens)

hist = seq_lens.hist()
hist.set_title("Sentence lengths histogram")

# truncate too long sentences and zero padding
_, long_sen_idx = trunc_pad(tokenized, max_input_size)
df.drop(df.iloc[long_sen_idx].index, inplace=True)

df_con = df[df["labels"] == 0]
df_exp = df[df["labels"] == 1]

# train:valid:test = 0.765:0.135:0.1
train_test_ratio = 0.90
train_valid_ratio = 0.85

df_con_train, df_con_test = train_test_split(df_con, train_size=train_test_ratio, random_state=1)
df_exp_train, df_exp_test = train_test_split(df_exp, train_size=train_test_ratio, random_state=1)

df_con_train, df_con_valid = train_test_split(df_con_train, train_size=train_valid_ratio, random_state=1)
df_exp_train, df_exp_valid = train_test_split(df_exp_train, train_size=train_valid_ratio, random_state=1)

df_train = pd.concat([df_con_train, df_exp_train], ignore_index=True, sort=False)
df_valid = pd.concat([df_con_valid, df_exp_valid], ignore_index=True, sort=False)
df_test = pd.concat([df_con_test, df_exp_test], ignore_index=True, sort=False)

# write processed data
destination_folder = "./"

df_train.to_csv(destination_folder+"train.csv", index=False)
df_valid.to_csv(destination_folder+"valid.csv", index=False)
df_test.to_csv(destination_folder+"test.csv", index=False)
