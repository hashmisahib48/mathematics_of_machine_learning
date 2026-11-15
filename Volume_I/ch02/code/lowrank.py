# Best low-rank copies of a photograph (Section 2.10).
import numpy as np
from sklearn.datasets import load_sample_image

img = load_sample_image("china.jpg").mean(axis=2)[::2, ::2]
m, n = img.shape
U, s, Vt = np.linalg.svd(img, full_matrices=False)
print("shape", img.shape, "  rank", np.linalg.matrix_rank(img))
for k in (2, 10, 40):
    Ak = (U[:, :k] * s[:k]) @ Vt[:k]               # Eq. 2.10-4
    err = np.linalg.norm(img - Ak) / np.linalg.norm(img)
    tail = np.sqrt((s[k:]**2).sum() / (s**2).sum())  # Eq. 2.10-5
    store = k * (m + n + 1) / (m * n)
    print(f"k = {k:2d}   error {err:.4f} = {tail:.4f}"
          f"   stored {100 * store:.1f}%")
