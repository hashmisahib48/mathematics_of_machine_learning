# FastICA on two mixed signals (the cocktail party with two voices)
import numpy as np
from sklearn.decomposition import FastICA, PCA

t = np.linspace(0, 8, 2000)
S = np.c_[np.sin(2 * t), 2 * (0.7 * t % 1) - 1]    # sine, sawtooth
S = (S - S.mean(axis=0)) / S.std(axis=0)           # unit variance
A = np.array([[1.0, 1.0], [0.5, 2.0]])             # mixing matrix
rng = np.random.RandomState(0)
X = S @ A.T + 0.05 * rng.standard_normal(S.shape)  # two microphones

def match(est):                                    # |corr|
    return np.abs(np.corrcoef(est.T, S.T)[:2, 2:])

ica = FastICA(2, random_state=0)
Se = ica.fit_transform(X)
print("ICA: |corr(estimate, source)|\n", match(Se).round(3))
print("PCA: |corr(score, source)|\n",
      match(PCA(2).fit_transform(X)).round(3))
print("mixing_ (columns match A up to sign and order)\n",
      ica.mixing_.round(3))
print("recovered scales:", Se.std(axis=0).round(3), " n_iter_ =",
      ica.n_iter_)
for fun in ("logcosh", "exp", "cube"):
    for alg in ("parallel", "deflation"):
        f = FastICA(2, fun=fun, algorithm=alg,
                    random_state=0).fit(X)
        r = match(f.transform(X)).max(axis=1).min()
        print(f"{fun:8s}{alg:10s} n_iter {f.n_iter_:3d}"
              f"  worst match {r:.4f}")
