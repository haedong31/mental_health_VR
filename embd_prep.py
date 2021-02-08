from pytorch_pretrained_bert import BertTokenizer
from bert_vocab_check import BertVocabCheck
from expand_contraction import expand_contraction
import re

## custom functions -----
def read_and_concat(self):
    base_dir = os.path.join(".", "data", "cookie_minimal_prep")
    if self.group == "control":
        paths = glob(os.path.join(base_dir, "control", "*.txt"))
    elif self.group == "experimental":
        paths = glob(os.path.join(base_dir, "dementia", "*.txt"))
    else:
        raise ValueError("Wrong group name")
    
    concat_utts = list()
    for path in paths:
        with open(path) as f:
            utt = f.read().splitlines()
            concat_utts.append(" ".join(utt))
    
    return concat_utts

## import data -----
con_utts = read_and_concat("control")
exp_utts = read_and_concat("experimental")

# check some words in BERT vocab
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
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
con_vocab = BertVocabCheck(con_utts)
exp_vocab = BertVocabCheck(exp_utts)

con_oov = con_vocab.oov
exp_oov = exp_vocab.oov

# top ten OOV words
print(list(con_oov.items())[:10])
print(list(exp_oov.items())[:10])

# exapand contractions
con_utts = expand_contraction(con_utts)
exp_utts = expand_contraction(exp_utts)

# remove 's from possesive constractions
const_pattern = re.compile("\'s")

const_pattern.sub("", con_utts)
const_pattern.sub("", exp_utts)
