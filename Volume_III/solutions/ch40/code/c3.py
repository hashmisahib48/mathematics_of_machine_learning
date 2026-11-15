# Problem 40C.3: component numbers are not cultivar numbers. Run: python3 c3.py
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../ch40/code'))
warnings.simplefilter('ignore', UserWarning)
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import load_wine
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
X = StandardScaler().fit_transform(X)
for s in (0, 3):
    gm = GaussianMixture(3, covariance_type="diag", init_params="random_from_data",
                         random_state=s, tol=1e-5, max_iter=500).fit(X)
    lab = gm.predict(X)
    T = np.array([[np.sum((y == c) & (lab == k)) for k in range(3)] for c in range(3)])
    r, c = linear_sum_assignment(-T)              # best matching of components to cultivars
    print(f"seed {s}: raw accuracy {np.mean(lab == y):.3f} (diagonal {np.trace(T)} of {len(y)})")
    print(T)
    print("  best matching cultivar -> component:", dict(zip(r.tolist(), c.tolist())),
          f" matched accuracy {T[r, c].sum() / len(y):.3f}"
          f"  ARI {adjusted_rand_score(y, lab):.3f}  mean log-lik {gm.score(X):.3f}")
    print("  after renaming:", (np.array([dict(zip(c, r))[k] for k in lab]) == y).mean().round(3))
