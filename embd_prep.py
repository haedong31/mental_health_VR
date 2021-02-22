from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
from bert_vocab_check import BertVocabCheck
from expand_contraction import expand_contraction
from glob import glob
import pandas as pd
import re
import os

## custom functions -----
def read_and_concat(group):
    base_dir = os.path.join(".", "data", "cookie_minimal_prep")
    if group == "control":
        paths = glob(os.path.join(base_dir, "control", "*.txt"))
    elif group == "experimental":
        paths = glob(os.path.join(base_dir, "dementia", "*.txt"))
    else:
        raise ValueError("Wrong group name")
    
    concat_utts = list()
    for path in paths:
        with open(path) as f:
            utt = f.read().splitlines()
            concat_utts.append(" ".join(utt))
    
    return concat_utts

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
con_utts = read_and_concat("control")
exp_utts = read_and_concat("experimental")

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

# conf = "./data/cookie_ebd_prep_con.txt"
# with open(conf, "wt") as f:
#     for u in con_utts:
#         f.write(u+"\n")

# expf = "./data/cookie_ebd_prep_exp.txt"
# with open(expf, "wt") as f:
#     for u in exp_utts:
#         f.write(u+"\n")

## words imputation with BERT -----
