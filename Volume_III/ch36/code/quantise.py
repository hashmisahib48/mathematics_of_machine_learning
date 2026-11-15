# Vector quantisation: a picture redrawn with K colours.
import numpy as np
from sklearn.cluster import KMeans

H, W = 90, 120                                 # a synthetic picture
yy, xx = np.mgrid[0:H, 0:W]
sky = np.stack([255 - 0.9 * yy, 150 + 0.4 * yy, 90 + 1.6 * yy], -1)
img = sky.copy()
sun = (xx - 80) ** 2 + (yy - 45) ** 2 < 14 ** 2
img[sun] = (255, 235, 120)
hill = yy > 60 + 12 * np.sin(xx / 14.0)
yh = yy[hill]
img[hill] = np.stack([30 + 0 * yh, 100 + 0.5 * yh, 55 + 0 * yh], -1)
rng = np.random.default_rng(0)
img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255)
pix = img.reshape(-1, 3)
print("pixels:", len(pix), " distinct colours:",
      len(np.unique(pix.round(), axis=0)))

print("K   bits/pixel  RMS error   share of 24-bit size")
quant = {}
for K in (2, 4, 8, 16, 32):
    km = KMeans(K, n_init=1, random_state=0).fit(pix)
    quant[K] = km.cluster_centers_[km.labels_].reshape(H, W, 3)
    bits = int(np.ceil(np.log2(K)))
    rms = np.sqrt(km.inertia_ / len(pix))      # root of J / n
    size = len(pix) * bits + K * 24            # indices + palette
    share = size / (len(pix) * 24)
    print(f"{K:2d}  {bits:10d}  {rms:9.2f}   {share:9.3f}")
