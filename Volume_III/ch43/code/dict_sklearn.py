# Dictionary learning and sparse-code denoising of 8 x 8 patches
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import SparseCoder
from sklearn.feature_extraction.image import (
    extract_patches_2d, reconstruct_from_patches_2d)

yy, xx = np.mgrid[:64, :64]                # synthetic 64 x 64 image
img = np.zeros((64, 64))
img[(yy - 20) ** 2 + (xx - 20) ** 2 < 100] = 1.0      # disc
img[8:12, 36:60] = 0.8                                # bar
img[(np.abs(yy - xx - 14) < 2) & (yy > 30)] = 0.6     # diagonal
r = np.hypot(yy - 46, xx - 44)
img[(r > 6) & (r < 9)] = 1.0                          # ring
rng = np.random.RandomState(0)
noisy = img + 0.3 * rng.standard_normal(img.shape)

P = extract_patches_2d(noisy, (8, 8)).reshape(-1, 64)  # 3249 rows
mu = P.mean(axis=1, keepdims=True)
Pc = P - mu                                # remove each patch mean
train = Pc[rng.choice(len(P), 2000, replace=False)]
dl = MiniBatchDictionaryLearning(
    32, alpha=1.0, batch_size=64, max_iter=30, fit_algorithm="lars",
    transform_algorithm="omp", random_state=0).fit(train)
H = dl.components_                         # 32 atoms of length 64
print("atoms", H.shape, " largest norm",
      np.linalg.norm(H, axis=1).max().round(4))
nz = (dl.transform(Pc[:200]) != 0).sum(axis=1).mean()
print("default omp: non-zeros per patch", nz)

rmse = lambda a: np.sqrt(np.mean((a - img) ** 2))
print("noisy image rmse", rmse(noisy).round(4))
for s in (1, 2, 4, 8):                     # atoms allowed per patch
    coder = SparseCoder(H, transform_algorithm="omp",
                        transform_n_nonzero_coefs=s)
    rec = (coder.transform(Pc) @ H + mu).reshape(-1, 8, 8)
    out = reconstruct_from_patches_2d(rec, img.shape)
    print(f"omp, {s} atoms per patch: rmse {rmse(out):.4f}")
blur = gaussian_filter(noisy, 1.0)
print("Gaussian blur, sigma 1:  rmse", rmse(blur).round(4))
