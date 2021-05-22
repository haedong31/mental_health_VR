from pathlib2 import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def gen_meta_df(data_dir, eye_dir, rp_dir, group):
    data_dir = Path(data_dir)
    eye_dir = Path(eye_dir)
    rp_dir = Path(rp_dir)
    
    # paths of raw data
    p = data_dir.glob('*')
    paths = [path for path in p if path.is_file()]
    files_wo_ext = [path.stem for path in paths]
    
    # paths of pseudo eye-tracking data
    eye_paths = [str(eye_dir/group/(file+'.png')) for file in files_wo_ext]
    
    # paths of recurrence plot data
    rp_paths = [str(rp_dir/group/(file+'.png')) for file in files_wo_ext]    

    if group == 'control':
        labels = [0]*len(paths)
    elif group == 'dementia':
        labels = [1]*len(paths)
    else:
        raise ValueError('Group name: control or experimental')
        
    return pd.DataFrame(data={'eye_path': eye_paths, 'rp_path': rp_paths, 'label': labels})
    
data_con_dir = Path('./data/DementiaBank/Pitt/Control/cookie')
data_exp_dir = Path('./data/DementiaBank/Pitt/Dementia/cookie')
eye_dir = Path('./data/pseudo_eyetracking')
rp_dir = Path('./data/audio')

train_test_ratio = 0.8

# healthy control group
con_meta = gen_meta_df(data_con_dir, eye_dir, rp_dir, group='control')
con_meta_train, con_meta_test = train_test_split(con_meta, train_size=train_test_ratio, random_state=1)

# experimental group (dementia)
exp_meta = gen_meta_df(data_exp_dir, eye_dir, rp_dir, group='dementia')
exp_meta_train, exp_meta_test = train_test_split(exp_meta, train_size=train_test_ratio, random_state=1)

# aggregate & shuffle
meta_train = pd.concat([con_meta_train, exp_meta_train], ignore_index=True, sort=False).sample(frac=1)
meta_test = pd.concat([con_meta_test, exp_meta_test], ignore_index=True, sort=False).sample(frac=1)

# save
meta_train.to_csv('./data/meta_train.csv', index=False)
meta_test.to_csv('./data/meta_test.csv', index=False)
