# Problem 42B.2: IncrementalPCA against PCA for several batch sizes
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
full = PCA(10, svd_solver="full").fit(Xtr)

def compare(data, b):
    ip = IncrementalPCA(10, batch_size=b).fit(data)
    cos = np.abs((ip.components_ * full.components_).sum(axis=1))
    err = np.abs(ip.explained_variance_
                 / full.explained_variance_ - 1).max()
    return cos, err

print("batch  min|cos|  worst component  max variance error")
for b in (50, 100, 200, 400, 800, 1347):
    cos, err = compare(Xtr, b)
    print(f"{b:5d}{cos.min():10.4f}{cos.argmin() + 1:10d}"
          f"{err:20.1e}")

# is the zigzag a property of the order of the samples?
print("\nten random orders: mean and sd of min|cos|")
rng = np.random.default_rng(0)
for b in (50, 100, 200, 400, 800):
    m = [compare(Xtr[rng.permutation(len(Xtr))], b)[0].min()
         for _ in range(10)]
    print(f"{b:5d}{np.mean(m):9.4f} +-{np.std(m):7.4f}")

lam = PCA(12, svd_solver="full").fit(Xtr).explained_variance_
print("\neigenvalues 8 to 12:", lam[7:].round(2))
print("ratios lam9/lam10 and lam10/lam11:",
      round(lam[8] / lam[9], 3), round(lam[9] / lam[10], 3))
