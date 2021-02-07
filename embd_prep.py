from pytorch_pretrained_bert import BertTokenizer
from bert_vocab_check import BertVocabCheck
from expand_contraction import expand_contraction


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

# 1st run
con_vocab = BertVocabCheck("control")
exp_vocab = BertVocabCheck("experimental")

print(list(con_vocab.oov.items())[:10])
print(list(exp_vocab.oov.items())[:10])

con_utts = con_vocab.utts
exp_utts = exp_vocab.utts

expand_contraction(con_utts[0])
