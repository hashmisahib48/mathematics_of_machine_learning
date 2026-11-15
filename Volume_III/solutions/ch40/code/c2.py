# Problem 40C.2: what is the fourth component of the diag fit of the wine data? Run: python3 c2.py
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../ch40/code'))
warnings.simplefilter('ignore', UserWarning)
import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
X = StandardScaler().fit_transform(X)
def fit(ct, K):                       # the fits of case_wine.py
    return GaussianMixture(K, covariance_type=ct, n_init=5, tol=1e-5, max_iter=500, random_state=0).fit(X)
def table(lab, K):
    return np.array([[np.sum((y == c) & (lab == k)) for k in range(K)] for c in range(3)])

for K in (3, 4):
    g = fit('diag', K); lab = g.predict(X)
    print(f"diag, K = {K}: BIC {g.bic(X):.0f}, mean log-lik {g.score(X):.3f}, ARI against the cultivars {adjusted_rand_score(y, lab):.3f}")
    T = table(lab, K)
    print("  rows: cultivars 0, 1, 2; columns: components, in order of size")
    print(T[:, np.argsort(-T.sum(0))])
    print("  component sizes", np.sort(np.bincount(lab, minlength=K))[::-1], " weights", np.sort(g.weights_)[::-1].round(3))
print("\nthe curves of the BIC for K = 1 ... 6 on the same data")
for ct in ('full', 'tied', 'diag', 'spherical'):
    b = [fit(ct, K).bic(X) for K in range(1, 7)]
    print(f"  {ct:10s}", " ".join(f"{v:6.0f}" for v in b), " minimum at K =", 1 + int(np.argmin(b)))
# why BIC likes the fourth component: log-likelihood gained against the penalty
g3, g4 = fit('diag', 3), fit('diag', 4)
p = X.shape[1]
q3, q4 = 2 + 2 * 3 * p, 3 + 2 * 4 * p
gain = len(X) * (g4.score(X) - g3.score(X))
print(f"\nparameters {q3} -> {q4}; log-likelihood gain {gain:.1f}; -2 gain = {-2 * gain:.1f},"
      f" penalty {(q4 - q3)} x ln 178 = {(q4 - q3) * np.log(len(X)):.1f}; BIC change {g4.bic(X) - g3.bic(X):.1f}")
