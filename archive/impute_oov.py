from pytorch_pretrained_bert import BertTokenizer,BertForMaskedLM
import torch
import numpy as np

# load pre-trained BERT model and tokenizer
with torch.no_grad():
    bert_model = BertForMaskedLM.from_pretrained('bert-base-cased')
    bert_model.eval()
bert_model = bert_model.to("cuda")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def bert_input(ds, oov_words, max_len):
    # convert data into input format of BERT
    for n, s in enumerate(ds):
        for word in oov_words:
            s = s.replace(word, "[MASK]")
        ds[n] = s

    max_len -= 2
    all_tokens = []
    longer = 0
    for s in ds:
        tokens4s = bert_tokenizer.tokenize(s)

        if len(tokens4s) > max_len:
            tokens4s = tokens4s[:max_len]
            longer += 1
        one_token = bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens4s + ["[SEP]"]) +[0] * (max_len - len(tokens4s))
        all_tokens.append(one_token)

    return np.array(all_tokens)

def idx_oov_words(x_batch, preds, offset):
    # index of masked words (oov words) 
    mask_idx = bert_tokenizer.vocab["[MASK]"]

    masked_idxs = np.argwhere(x_batch == mask_idx)
    locs = masked_idxs[:, 0] + offset

    pt_preds = np.argmax(preds[x_batch == mask_idx], axis=-1)
    loc2pred = {}
    for loc, pred in zip(locs, pt_preds):
        loc2pred[loc] = loc2pred.get(loc, []) + [pred]

    return loc2pred

def predict_OOV(texts, max_len, words_to_replace, batch_size=32):
    ids = bert_input(texts, oov_words, max_len)
    data_loader = torch.utils.data.DataLoader(torch.tensor(ids), batch_size=batch_size, shuffle=False)
    
    idx2prediction = {}
    for i, x_batch  in tqdm(enumerate(data_loader)):
        text_preds = model(x_batch.to('cuda'),
                           token_type_ids=None,
                           attention_mask=None,
                           masked_lm_labels=None).data.cpu().numpy()
        batch_preds = get_location2prediction(x_batch.numpy(), text_preds, offset = i * batch_size)
        idx2prediction.update(batch_preds)

    return idx2prediction

idx2prediction = predict_OOV(train['comment_text'].values[:200],
                    max_seq_length=220,
                    words_to_replace=oov_words,
                    batch_size=32)

print([bert_tokenizer.ids_to_tokens[idx] for idx in idx2prediction[117]])
print(train['comment_text'].iloc[117])
