from pytorch_pretrained_bert import BertTokenizer
import os
from glob import glob


class BertVocabCheck:
    def __init__(self, group):
        self.group = group
        self.utts = list()
        self.freq_vocab = dict()
        self.oov = dict()

        self.run()

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
        self.utts = concat_utts

    def word_counter(self):
        for utt in self.utts:
            words = utt.split()
            for word in words:
                if word in self.freq_vocab:
                    self.freq_vocab[word] += 1
                else:
                    self.freq_vocab[word] = 1

    def vocab_bert_check(self):
        oov = dict()
        num_embd = 0
        freq_embd = 0
        freq_oov = 0

        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        for word in self.freq_vocab:
            if word in tokenizer.vocab:
                num_embd += 1
                freq_embd += self.freq_vocab[word]
            else:
                oov[word] = self.freq_vocab[word]
                freq_oov += self.freq_vocab[word]
        
        print(f"{self.group} group")
        print(f"% of words in BERT vocab: {num_embd/len(self.freq_vocab):.2%}")
        print(f"% of word counts in BERT vocab: {freq_embd/(freq_embd+freq_oov):.2%}")
        self.oov = dict(sorted(oov.items(), key=lambda x: x[1], reverse=True))

    def run(self):
        self.read_and_concat()
        self.word_counter()
        self.vocab_bert_check()
