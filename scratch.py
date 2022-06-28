from pathlib import Path
import pylangacq

import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# import torch
# import torchaudio
# import torchaudio.transforms as transforms

# from bert_vocab_check import BertVocabCheck

##### text data basic statistics -----
hc = pylangacq.read_chat('./data/DementiaBank/Pitt/Control/cookie') # healthy control
ad = pylangacq.read_chat('./data/DementiaBank/Pitt/Dementia/cookie') # Alzheimer's disease

print(f'Number of participants (Control): {hc.n_files()}')
print(f'Number of participants (Dementia): {ad.n_files()}')

numu_hc = []
numw_ad = []
for n,f in enumerate(hc):
    print(f"Participant {n+1}'s utterances {len(f.utterances(participants='PAR'))}")
    print(f"Participant {n+1}'s words {len(f.words(participants='PAR'))}")

##### check meta-data files -----
data_con_dir = Path('./data/DementiaBank/Pitt/Control/cookie')
data_exp_dir = Path('./data/DementiaBank/Pitt/Dementia/cookie')
eye_dir = Path('./data/pseudo_eyetracking')
rp_dir = Path('./data/audio')

eye_trn = pd.read_csv(str(eye_dir/'train_meta.csv'))
eye_trn['label'].value_counts()
eye_trn['path'].head()

rp_trn = pd.read_csv(str(rp_dir/'meta_train.csv'))
rp_trn['label'].value_counts()
rp_trn['path'].head()


agg = pd.read_csv('./data/meta_train.csv')
agg['label'].value_counts()
agg.head()


p1 = data_con_dir.glob('*')
f1 = [x.stem for x in p1]

p2 = (eye_dir/'control').glob('*.png')
f2 = [x.stem for x in p2]

p3 = (rp_dir/'control').glob('*.png')
f3 = [x.stem for x in p3]

def common_elts(a,b):
    seta = set(a)
    setb = set(b)
    
    intersec = (seta & setb)
    
    if intersec:
        return intersec
    else:
        print('No common elements')

u1 = common_elts(f1,f2)
u2 = common_elts(f2,f3)


##### words counts -----
def sort_save_freq_vocab(freq_vocab, save_path):
    freq_vocab = dict(sorted(freq_vocab.items(), key=lambda x: x[1], reverse=True))
    k = list(freq_vocab.keys())
    v = list(freq_vocab.values())
    
    df = pd.DataFrame(data={"word": k, "count": v})
    df.to_excel(save_path, index=False)
    
df = pd.read_csv("./data/text/cookie_ebd_prep.csv", index_col=0)
df_con = df[df["labels"] == 0]
df_exp = df[df["labels"] == 1]

con_vocab = BertVocabCheck(df_con["utts"])
exp_vocab = BertVocabCheck(df_exp["utts"])

sort_save_freq_vocab(con_vocab.freq_vocab, './data/pseudo_eyetracking/con_word_freq.xlsx')
sort_save_freq_vocab(exp_vocab.freq_vocab, './data/pseudo_eyetracking/exp_word_freq.xlsx')

# frequency vocab
con_freq_vocab = con_vocab.freq_vocab
exp_freq_vocab = exp_vocab.freq_vocab

# sort by dictionary values
con_freq_vocab = dict(sorted(con_freq_vocab.items(), key=lambda x: x[1], reverse=True))
exp_freq_vocab = dict(sorted(exp_freq_vocab.items(), key=lambda x: x[1], reverse=True))

##### audio dataset -----
config_dict = {"num_epochs": 6, "batch_size": 8, "drop_out": 0.25,
    "base_lr": 0.005, "num_mel_filters": 2048, "resample_freq": 22050}

audio_file_paths = []
labels = []
num_mel_filters = config_dict.get("num_mel_filters")
resample_freq = config_dict.get("resample_freq")

meta_data = pd.read_csv("./data/audio/meta_train.csv")
for i in range(0, len(meta_data)):
    audio_file_paths.append(meta_data.iloc[i,0])
    labels.append(meta_data.iloc[i,1])

# read data and resample
num_obs = 3
mel_spectros = []
mel_spectros_db = []
for idx in range(0, num_obs):
    sound_data, sample_rate = torchaudio.load(audio_file_paths[idx])
    
    resampler = transforms.Resample(
        orig_freq=sample_rate, 
        new_freq=resample_freq)
    sound_data = resampler(sound_data)
    
    # mono channel
    sound_data = torch.mean(sound_data, dim=0, keepdim=True)
    
    # spectrogram
    mel_spectro_transform = transforms.MelSpectrogram(
        sample_rate=resample_freq, 
        n_mels=num_mel_filters)
    decibel_transform = transforms.AmplitudeToDB()
    
    mel_spectro = mel_spectro_transform(sound_data)
    mel_spectro_db = decibel_transform(mel_spectro)
    
    mel_spectros.append(mel_spectro)
    mel_spectros_db.append(mel_spectro_db)
    
plt.figure()
plt.imshow(mel_spectros[2].squeeze().numpy(), cmap="hot")
