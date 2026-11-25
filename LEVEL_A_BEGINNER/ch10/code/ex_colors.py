# Reducing a photograph to 16 colors.
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image

img = load_sample_image("china.jpg") / 255.0     # 427 x 640 x 3
P = img.reshape(-1, 3)                           # one row per pixel
rng = np.random.default_rng(0)
sample = P[rng.choice(len(P), 2000, replace=False)]
m = KMeans(n_clusters=16, n_init=4, random_state=0).fit(sample)
lab = m.predict(P)                               # nearest of 16 colors
Q = m.cluster_centers_[lab]
print("pixels", len(P), " distinct colors", len(np.unique(P, axis=0)))
print("root-mean-square error per channel",
      round(float(np.sqrt(np.mean((P - Q) ** 2))), 4))
print("bits per pixel: 24 ->", int(np.log2(16)))
