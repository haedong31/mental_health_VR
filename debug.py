import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from pathlib import Path

def ifs(states,k,alpha):
    x = []
    y = []
    x.append(0)
    y.append(0)

    for i in range(len(states)):
        x.append(alpha*x[i]+np.cos(states[i]*2*np.pi/k))
        y.append(alpha*y[i]+np.sin(states[i]*2*np.pi/k))
    
    ifs_df = pd.DataFrame({'xaddress':x, 'yaddress':y})
    ifs_df = ifs_df.drop(index=0) # drop first row

    return ifs_df

def heterorecurrence1(ifs_coord,sig,order):
    idx = np.where(sig == order)[0]
    if np.size(idx) != 0:
        rr = np.power(len(idx),2) / np.power(len(sig),2)
        
        rmx = cerecurr_y(ifs_coord.iloc[idx,:])
        trir = np.triu(rmx,k=1)
        flatr = trir.flatten()
        flatr = flatr[flatr != 0]
        
        if np.size(flatr) == 0:
            rent = 0            
            rmean = 0
        else:
            hist_count = np.histogram(flatr,bins='auto')[0]
            prob = hist_count/np.sum(hist_count)
            prob = prob[prob != 0]
            rent = np.sum(prob*(-np.log(prob)))
            rmean = np.mean(flatr)            
    else:
        rr = 0
        rent = 0
        rmean = 0
    return (rr,rent,rmean)
            

def heterorecurrence2(ifs_coord,sig,o1,o2):
    idx1 = np.where(sig == o1)[0]
    if np.size(idx1) != 0:
        idx2 = np.where(sig.iloc[idx1] == o2)[0]
        if np.size(idx2) != 0:
            rr = np.power(len(idx2),2) / np.power(len(sig),2)
            
            rmx = cerecurr_y(ifs_coord.iloc[idx2,:])
            trir = np.triu(rmx,k=1)
            flatr = trir.flatten()
            flatr = flatr[flatr != 0]
            
            if np.size(flatr) == 0:
                rent = 0
                rmean = 0
            else:
                hist_count = np.histogram(flatr,bins='auto')[0]
                prob = hist_count/np.sum(hist_count)
                prob = prob[prob != 0]
                rent = np.sum(prob*(-np.log(prob)))
                rmean = np.mean(flatr)
        else:
            rr = 0
            rent = 0
            rmean = 0
    else:
        rr = 0
        rent = 0
        rmean = 0
    return (rr,rent,rmean)
    
def cerecurr_y(ifs_coord):
    n = ifs_coord.shape[0]
    buff = np.empty([n,n])
    
    for i in range(n):
        for j in range(n):
            d = np.linalg.norm(ifs_coord.iloc[i,:]-ifs_coord.iloc[j,:])
            buff[i,j] = d
            buff[j,i] = d
    return buff

def hra_features(p,num_tokens,alpha):
    file_paths = Path(p).glob('*.csv')
    fsize = len(list(file_paths))
    file_paths = Path(p).glob('*.csv')
    
    hr_feat_mx = np.empty((0,3*num_tokens+num_tokens*num_tokens*3))
    for n,f, in enumerate(file_paths):
        if (n%5) == 0:
            print(f'Progress: {np.round(n/fsize,2)} %')
        
        sig = pd.read_csv(f)
        sig = sig.x
        ifs_coord = ifs(sig,num_tokens,alpha)
        
        rr1 = []
        rent1 = []
        rmean1 = []
        hr_feat2_list = []
        
        # 1st order
        for i in range(1,num_tokens+1):
            hr_feat1 = heterorecurrence1(ifs_coord,sig,i)
            rr1.append(hr_feat1[0])
            rent1.append(hr_feat1[1])
            rmean1.append(hr_feat1[2])
            
            rr2 = []
            rent2 = []
            rmean2 = []
            # 2nd order
            for j in range(1,num_tokens+1):
                hr_feat2 = heterorecurrence2(ifs_coord,sig,i,j)
                rr2.append(hr_feat2[0])
                rent2.append(hr_feat2[1])
                rmean2.append(hr_feat2[2])
            hr_feat2_list = hr_feat2_list+rr2+rent2+rmean2
        row = rr1+rent1+rmean1+hr_feat2_list
        np.append(hr_feat_mx,np.array([row]),axis=0)
    return hr_feat_mx
        
phc = './data/hra/tf-idf_control/'
pad = './data/hra/dementia'
num_tokens = 32
alpha = 0.04

feat_mx_hc = hra_features(phc, num_tokens, alpha)
feat_mx_ad = hra_features(pad, num_tokens, alpha)
