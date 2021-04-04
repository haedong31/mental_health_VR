import os
from glob import glob
from pydub import AudioSegment

import pandas as pd

from sklearn.model_selection import train_test_split

def multiple_mp3_to_wav(dir):
    src_paths =  glob(dir + "/*.mp3")

    for src in src_paths:
        wo_ext = os.path.splitext(src)[0]
        out_path =  wo_ext + ".wav"
        
        sound = AudioSegment.from_mp3(src)
        sound.export(out_path, format="wav")

## conver mp3 to wav
multiple_mp3_to_wav("./data/audio/control/cookie")
multiple_mp3_to_wav("./data/audio/dementia/cookie")

## split into train, valid, and test dataset -----
# train:valid:test = 0.765:0.135:0.1
train_test_ratio = 0.90
train_valid_ratio = 0.85

# control group
con_paths = glob("./data/audio/control/cookie/*.wav")
con_meta = pd.DataFrame(data={"file_paths": con_paths, "class": [0]*len(con_paths)})
con_meta_train, con_meta_test = train_test_split(con_meta, train_size=train_test_ratio, random_state=1)
con_meta_train, con_meta_valid = train_test_split(con_meta_train, train_size=train_valid_ratio, random_state=1)

# experimental group (dementia)
exp_paths = glob("./data/audio/dementia/cookie/*.wav")
exp_meta = pd.DataFrame(data={"file_paths": exp_paths, "class": [1]*len(exp_paths)})
exp_meta_train, exp_meta_test = train_test_split(exp_meta, train_size=train_test_ratio, random_state=1)
exp_meta_train, exp_meta_valid = train_test_split(exp_meta_train, train_size=train_valid_ratio, random_state=1)

# aggregate & shuffle
meta_train = pd.concat([con_meta_train, exp_meta_train], ignore_index=True, sort=False).sample(frac=1)
meta_valid = pd.concat([con_meta_valid, exp_meta_valid], ignore_index=True, sort=False).sample(frac=1)
meta_test = pd.concat([con_meta_test, exp_meta_test], ignore_index=True, sort=False).sample(frac=1)

# save
meta_train.to_csv("./data/audio/meta_train.csv", index=False)
meta_valid.to_csv("./data/audio/meta_valid.csv", index=False)
meta_test.to_csv("./data/audio/meta_test.csv", index=False)
