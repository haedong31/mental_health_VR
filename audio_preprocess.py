import os
from pathlib2 import Path
from glob import glob
from pydub import AudioSegment

import pandas as pd
from sklearn.model_selection import train_test_split

def multiple_mp3_to_wav(source_dir, out_dir):
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    src_paths =  glob(source_dir + "/*.mp3")
    for src in src_paths:
        src = Path(src)
        
        fname = src.name
        fname = os.path.splitext(fname)[0]
        fname = fname + '.wav'
        
        sound = AudioSegment.from_mp3(src)
        sound.export(out_dir/fname, format="wav")

##### convert mp3 to wav -----
multiple_mp3_to_wav("./data/audio_mp3/control/cookie", './data/audio/control')
multiple_mp3_to_wav("./data/audio_mp3/dementia/cookie", './data/audio/dementia')

##### recurrence plot image augmentation -----


##### split into train, valid, and test dataset -----
# train:valid:test = 0.765:0.135:0.1
train_test_ratio = 0.90
train_valid_ratio = 0.85

# control group
base_dir = Path('./data/audio')
con_paths = glob(str(base_dir/'control'/'*.png'))
con_meta = pd.DataFrame(data={"path": con_paths, "label": [0]*len(con_paths)})
con_meta_train, con_meta_test = train_test_split(con_meta, train_size=train_test_ratio, random_state=1)
con_meta_train, con_meta_valid = train_test_split(con_meta_train, train_size=train_valid_ratio, random_state=1)

# experimental group (dementia)
exp_paths = glob(str(base_dir/'dementia'/'*.png'))
exp_meta = pd.DataFrame(data={"path": exp_paths, "label": [1]*len(exp_paths)})
exp_meta_train, exp_meta_test = train_test_split(exp_meta, train_size=train_test_ratio, random_state=1)
exp_meta_train, exp_meta_valid = train_test_split(exp_meta_train, train_size=train_valid_ratio, random_state=1)

# aggregate & shuffle
meta_train = pd.concat([con_meta_train, exp_meta_train], ignore_index=True, sort=False).sample(frac=1)
meta_valid = pd.concat([con_meta_valid, exp_meta_valid], ignore_index=True, sort=False).sample(frac=1)
meta_test = pd.concat([con_meta_test, exp_meta_test], ignore_index=True, sort=False).sample(frac=1)

# save
meta_train.to_csv(str(base_dir/'meta_train.csv'), index=False)
meta_valid.to_csv(str(base_dir/'meta_valid.csv'), index=False)
meta_test.to_csv(str(base_dir/'meta_test.csv'), index=False)
