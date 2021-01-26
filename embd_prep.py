from pytorch_pretrained_bert import BertTokenizer
import os
from glob import glob


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

def word_counter(utterances):
    freq_vocab = dict()
    for utt in utterances:
        words = utt.split()
        for word in words:
            if word in freq_vocab:
                freq_vocab[word] += 1
            else:
                freq_vocab[word] = 1
    return freq_vocab

def vocab_bert_check(vocab):
    oov = dict()
    num_embd = 0
    freq_embd = 0
    freq_oov = 0

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    for word in vocab:
        if word in tokenizer.vocab:
            num_embd += 1
            freq_embd += vocab[word]
        else:
            oov[word] = vocab[word]
            freq_oov += vocab[word]
    
    print(f"% of words in BERT vocab: {num_embd/len(tokenizer.vocab):.2%}")
    print(f"% of word counts in BERT vocab: {freq_embd/(freq_embd+freq_oov):.2%}")
    sorted_oov = dict(sorted(oov.items(), key=lambda x: x[1], reverse=True))

    return sorted_oov

## main -----
cookie_con = read_and_concat("control")
cookie_exp = read_and_concat("experimental")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# 'she' in tokenizer.vocab # True
# 'I' in tokenizer.vocab # False
# 'he' in tokenizer.vocab # True
# 'is' in tokenizer.vocab # True
# 'are' in tokenizer.vocab # True
# 'overflow' in tokenizer.vocab # False
# '.' in tokenizer.vocab # True
# '?' in tokenizer.vocab # True
'flowing' in tokenizer.vocab

# 1st run
freq_vocab_con = word_counter(cookie_con)
freq_vocab_exp = word_counter(cookie_exp)

print("Control group")
oov_con = vocab_bert_check(freq_vocab_con)

print("Experimental group")
oov_exp = vocab_bert_check(freq_vocab_exp)

