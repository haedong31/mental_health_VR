import os
from pathlib import Path

import cv2
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from bb_label_mapping import bb_label_mapping

eyetrack_dir = Path('./data/pseudo_eyetracking')
text_dir = Path('./data/text')

# Boston Cookie Theft picture
# cookie = cv2.cvtColor(cv2.imread(os.path.join(eyetrack_dir, 'cookie.jpg')), cv2.COLOR_BGR2RGB)
cookie = Image.open(os.path.join(eyetrack_dir, 'cookie.jpg'))
cookie = cookie.convert('RGB')

# bounding-box labels on the Boston Cookie picture
bb_label_df = pd.read_csv(os.path.join(eyetrack_dir, 'labels_boston_cookie.csv'), header=None)
gaze_words = bb_label_mapping.keys()

# transcripts
text_df = pd.read_csv(os.path.join(text_dir, 'cookie_ebd_prep.csv'))

def fill_box(im, label):
    print('empty')

for idx, row in text_df.iterrows():
    pass

label1 = bb_label_df.iloc[0]
plt.imshow(cookie)
plt.gca().add_patch(Rectangle((label1[1],label1[2]), label1[3], label1[4], 
                              color='red', fill=False, lw=3))

row = text_df.iloc[0]
utt = row['utts']

# fill box
for word in utt.split():
    print(word)
    if word in gaze_words:
        bb_labels = bb_label_mapping.get(word)
        for bb_label in bb_labels:
            idx = bb_label_df[0].index[bb_label_df[0] == bb_label]
            bb_label_info = bb_label_df.loc[idx].values.flatten().tolist()
            
            # need to implement code filling bounding box using info in bb_label_info