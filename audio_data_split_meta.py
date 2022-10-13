import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


data_dir1 = Path("./data/audio/control_wav")
data_dir2 = Path("./data/audio/dementia_wav")

fpaths1 = data_dir1.glob("*.wav")
fpaths2 = data_dir2.glob("*.wav")

p = []
l = []
for f in fpaths1:
    p.append(f.stem)
    l.append(0)

for f in fpaths2:
    p.append(f.stem)
    l.append(1)

df = pd.DataFrame(data={"file":p, "label":l})

# train:valid:test = 0.765:0.135:0.1
train_test_ratio = 0.9
train_valid_ratio = 0.85

# test
meta_train, meta_test = train_test_split(df, train_size=train_test_ratio)

# valid
meta_train, meta_valid = train_test_split(meta_train, train_size=train_valid_ratio)

meta_train.to_csv("./data/audio/meta_train.csv", index=False)
meta_valid.to_csv("./data/audio/meta_valid.csv", index=False)
meta_test.to_csv("./data/audio/meta_test.csv", index=False)
