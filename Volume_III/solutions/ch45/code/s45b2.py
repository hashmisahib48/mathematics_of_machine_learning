# Problem 45B.2: Nystroem landmarks, random images or k-means centres
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def rbf(A, B, g):                          # Eq. 21.3-7 (Volume II)
    d2 = (A**2).sum(1)[:, None] + (B**2).sum(1)[None] - 2 * A @ B.T
    return np.exp(-g * np.maximum(d2, 0))

def fit(L, g):                             # W^(-1/2), landmarks L
    U, s, Vt = np.linalg.svd(rbf(L, L, g))
    s = np.clip(s, 1e-12, None)
    return (U / np.sqrt(s)) @ Vt           # Eq. 45.4-7

def transform(X, L, g, Wih):
    return rbf(X, L, g) @ Wih.T
X, y = load_digits(return_X_y=True)
X = train_test_split(X, y, test_size=0.25, random_state=0,
                    stratify=y)[0] / 16
g = 0.1105; K = rbf(X, X, g)
print("seed   error: rand  k-m   trace/n: rand  k-m    d2: rand  k-m")
res = {"random": [], "k-means": []}
for seed in range(5):
    pick = np.random.RandomState(seed).permutation(len(X))[:50]
    km = KMeans(50, n_init=1, random_state=seed).fit(X)
    sets = {"random": X[pick], "k-means": km.cluster_centers_}
    row = []
    for name, L in sets.items():
        Z = transform(X, L, g, fit(L, g))
        E = K - Z @ Z.T
        d2 = ((X[:, None] - L[None])**2).sum(2).min(1).mean()  # to L
        res[name].append(np.linalg.norm(E) / np.linalg.norm(K))
        row.append((np.trace(E) / len(X), d2))
    er, ek = res["random"][-1], res["k-means"][-1]
    print(f"{seed:3d}  {er:12.4f} {ek:.4f}  {row[0][0]:12.4f} "
          f"{row[1][0]:.4f}  {row[0][1]:10.2f} {row[1][1]:.2f}")
for name in res:
    print(name, "mean relative error: %.4f" % np.mean(res[name]))
