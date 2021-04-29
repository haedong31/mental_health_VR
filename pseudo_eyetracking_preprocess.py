import os
from pathlib import Path
from glob import glob
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split

from bb_label_mapping import bb_label_mapping

##### generate pseudo eye-tracking data -----
eyetrack_dir = Path('./data/pseudo_eyetracking')
text_dir = Path('./data/text')

# Boston Cookie Theft picture
cookie = Image.open(os.path.join(eyetrack_dir, 'cookie.jpg'))
cookie = cookie.convert('RGB')
cookie_width, cookie_height = cookie.size

# bounding-box labels on the Boston Cookie picture
bb_label_df = pd.read_csv(os.path.join(eyetrack_dir, 'labels_boston_cookie.csv'), header=None)
gaze_words = bb_label_mapping.keys()

# transcripts
text_df = pd.read_csv(os.path.join(text_dir, 'cookie_ebd_prep.csv'))

# coloring "gazing boxes" on the Boston Cookie picture
def fill_box(utt, num_boxes=5):
    # create graphical objects
    fig, ax = plt.subplots()
    
    # draw the cookie image
    p = ax.imshow(cookie)

    gaze_words_insitu = []
    len_gaze_words_insitu = 0
    words = utt.split()
    for n, word in enumerate(words):
        if word in gaze_words:
            # save word for exception rules later
            gaze_words_insitu.append(word)
            len_gaze_words_insitu += 1

            bb_labels = bb_label_mapping.get(word)
            for bb_label in bb_labels:
                # exception 1: cookie jar (prevent double counting)
                if word == 'jar':
                    # check the previous gazing word
                    if (gaze_words_insitu[(len_gaze_words_insitu-1)-1] == 'cookie' or 
                        gaze_words_insitu[(len_gaze_words_insitu-1)-1] == 'cookies'):
                        continue
                # exception 2: lady (girl or mother)
                elif word == 'lady':
                    if n == 0:
                        pass
                    if words[n-1] == 'young' or words[n-1] == 'little':
                        bb_label = 'girl'
                        gaze_words_insitu[(len_gaze_words_insitu-1)] = 'young lady'
                # exception 3: she (girl or mother)
                elif word == 'she':
                    # find the nearest word referring 'she'
                    she_candidates = ['lady', 'young lady', 'girl', 'girls']
                    she_ref = ''
                    for gw in gaze_words_insitu:
                        if gw in she_candidates:
                            she_ref = gw
                    
                    if she_ref == '':
                        bb_label = bb_label_mapping.get('she')[0] # girl
                    else:
                        bb_label = bb_label_mapping.get(she_ref)[0]
                    
                idx = bb_label_df[0].index[bb_label_df[0] == bb_label]
                bb_label_info = bb_label_df.loc[idx].values.flatten().tolist()

                # bounding-box information            
                xmin = bb_label_info[1]
                ymin = bb_label_info[2]
                w = bb_label_info[3]
                h = bb_label_info[4]
                
                # center of bounding box
                xcenter = xmin + np.floor(w/2)
                ycenter = ymin + np.floor(h/2)

                # add filling boxes
                for _ in range(num_boxes):
                    scaler = np.random.uniform(0,1,1)[0]
                    sw = scaler*w
                    sh = scaler*h
                                    
                    xprime = xcenter - np.floor(sw/2)
                    yprime = ycenter - np.floor(sh/2)
                    
                    p = ax.add_patch(Rectangle(
                            (xprime,yprime), sw, sh, color='red', fill=True, alpha=0.1))
    return fig, p

for idx, row in text_df.iterrows():
    fig, p = fill_box(row['utts'])
    text_path = row['paths']
    
    file_name = os.path.basename(text_path)
    file_name = os.path.splitext(file_name)[0] + '.png'
    
    if row['labels'] == 0:
        os.path.join(eyetrack_dir, 'control', file_name)
        fig.savefig(os.path.join(eyetrack_dir, 'control', file_name))
    elif row['labels'] == 1:
        fig.savefig(os.path.join(eyetrack_dir, 'dementia', file_name))
    else:
        raise ValueError('save path does not exist')
    plt.clf()
    
##### train-test split -----
con_paths = glob(os.path.join(eyetrack_dir, 'control', '*.png'))
exp_paths = glob(os.path.join(eyetrack_dir, 'dementia', '*.png'))

con_meta_df = pd.DataFrame(data={'path':con_paths, 'label':[0]*len(con_paths)})
exp_meta_df = pd.DataFrame(data={'path':exp_paths, 'label':[1]*len(exp_paths)})

# train:valid:test = 0.765:0.135:0.1
train_test_ratio = 0.9
train_valid_ratio = 0.85

# test
con_meta_train, con_meta_test = train_test_split(con_meta_df, train_size=train_test_ratio)
exp_meta_train, exp_meta_test = train_test_split(exp_meta_df, train_size=train_test_ratio)

# valid
con_meta_train, con_meta_valid = train_test_split(con_meta_train, train_size=train_valid_ratio)
exp_meta_train, exp_meta_valid = train_test_split(exp_meta_train, train_size=train_valid_ratio)

meta_train = pd.concat([con_meta_train,exp_meta_train], ignore_index=True, sort=False)
meta_train = meta_train.sample(frac=1)
meta_valid = pd.concat([con_meta_valid,exp_meta_valid], ignore_index=True, sort=False)
meta_valid = meta_valid.sample(frac=1)
meta_test = pd.concat([con_meta_test,exp_meta_test], ignore_index=True, sort=False)
meta_test = meta_test.sample(frac=1)

# save
meta_train.to_csv(os.path.join(eyetrack_dir, 'tain_meta.csv'), index=False)
meta_valid.to_csv(os.path.join(eyetrack_dir, 'valid_meta.csv'), index=False)
meta_test.to_csv(os.path.join(eyetrack_dir, 'test_meta.csv'), index=False)
