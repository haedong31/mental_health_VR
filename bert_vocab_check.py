from transformers import BertTokenizer

class BertVocabCheck:
    def __init__(self, data):
        self.utts = data
        self.freq_vocab = dict()
        self.oov = dict()
        self.run()

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

        tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
        for word in self.freq_vocab:
            if word in tokenizer.vocab:
                num_embd += 1
                freq_embd += self.freq_vocab[word]
            else:
                oov[word] = self.freq_vocab[word]
                freq_oov += self.freq_vocab[word]
        
        print(f"% of words in BERT vocab: {num_embd/len(self.freq_vocab):.2%}")
        print(f"% of word counts in BERT vocab: {freq_embd/(freq_embd+freq_oov):.2%}")
        self.oov = dict(sorted(oov.items(), key=lambda x: x[1], reverse=True))

    def run(self):
        self.word_counter()
        self.vocab_bert_check()
