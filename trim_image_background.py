from pathlib2 import Path
from PIL import Image
from PIL import ImageOps

import pandas as pd
import matplotlib.pyplot as plt

def crop_box(img):
    inverted = img.convert('RGB')
    inverted = ImageOps.invert(inverted)
    
    img_box = inverted.getbbox()
    cropped = img.crop(img_box)
    
    return cropped
    
meta_df = pd.read_csv('./data/meta_train.csv')

eye_path1 = Path(meta_df.iloc[0]['eye_path'])
eye_path2 = Path(meta_df.iloc[1]['eye_path'])

rp_path1 = Path(meta_df.iloc[0]['rp_path'])
rp_path2 = Path(meta_df.iloc[1]['rp_path'])

eye_img1 = Image.open(str(eye_path1))
plt.imshow(eye_img1)
eye_img2 = Image.open(str(eye_path2))
plt.imshow(eye_img2)

rp_img1 = Image.open(str(rp_path1))
plt.imshow(rp_img1)
rp_img2 = Image.open(str(rp_path2))
plt.imshow(rp_img2)

cropped1 = crop_box(eye_img1)
plt.imshow(cropped1)
cropped2 = crop_box(eye_img2)
plt.imshow(cropped2)
cropped3 = crop_box(rp_img1)
plt.imshow(cropped3)
cropped4 = crop_box(rp_img2)
plt.imshow(cropped4)
