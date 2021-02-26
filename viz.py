from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import seaborn as sns

# load sentence emberddr
embedder = SentenceTransformer("paraphrase-distilroberta-base-v1")

# load data
with open("./data/cookie_ebd_prep_con.txt", "rt") as f:
    con_utts = f.read().splitlines()
    
with open("./data/cookie_ebd_prep_exp.txt", "rt") as f:
    exp_utts = f.read().splitlines()

# sentence embedding    
utts = con_utts + exp_utts
ebd_utts = embedder.encode(utts)

# dimensionality reduction by PCA
pca_dim = 15
pca = PCA(n_components=pca_dim)
pca_decomp = pca.fit_transform(ebd_utts)
print(f"Explained variance with {pca_dim}-dim: {np.sum(pca.explained_variance_ratio_):.4f}")

# visualization by t-SNE
tsne = TSNE(n_components=2, perplexity=40, n_iter=50000)
tsne_rslt = tsne.fit_transform(ebd_utts)

y = [0]*len(con_utts) + [1]*len(exp_utts)
df = pd.DataFrame(data={"x1": tsne_rslt[:,0], "x2": tsne_rslt[:,1], "y": y})
sns.scatterplot(
    x="x1", y="x2", 
    hue="y", palette=sns.color_palette("hls", 2),
    data=df,
    legend=["control", "dementia"])