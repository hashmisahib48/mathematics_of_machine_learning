# FastMCD (n <= 500) and the reweighting step, from scratch.
import numpy as np
from scipy.stats import chi2
from sklearn.covariance import MinCovDet

def fit_subset(X, H):                       # mean, covariance of H
    return X[H].mean(0), np.cov(X[H].T, bias=True)

def d2(X, m, S):                            # Eq. 46.3-5
    return np.einsum("ij,jk,ik->i", X - m, np.linalg.pinv(S), X - m)

def cstep(X, H, h):                         # Eq. 46.6-3
    m, S = fit_subset(X, H)
    return np.argsort(d2(X, m, S))[:h]

def c_alpha(p, a):                          # Eq. 46.6-5
    return a / chi2.cdf(chi2.ppf(a, p), p + 2)

def logdet(X, H):
    return np.linalg.slogdet(fit_subset(X, H)[1])[1]

def my_mcd(X, seed):
    n, p = X.shape
    h = int(np.ceil((n + p + 1) / 2))
    rs = np.random.RandomState(seed)
    cand = []
    for _ in range(30):                     # 30 starts, 2 C-steps
        H = rs.permutation(n)[:h]
        for _ in range(2):
            H = cstep(X, H, h)
        cand.append((logdet(X, H), H))
    cand.sort(key=lambda t: t[0])
    best = (np.inf, None)
    for _, H in cand[:10]:                  # best 10 to convergence
        for _ in range(30):
            H2 = cstep(X, H, h)
            if set(H2) == set(H):
                break
            H = H2
        best = min(best, (logdet(X, H), H), key=lambda t: t[0])
    m, S = fit_subset(X, best[1])           # raw estimate
    c = c_alpha(p, h / n)
    keep = d2(X, m, S) / c < chi2.isf(0.025, p)   # reweighting
    m2, S2 = fit_subset(X, keep)
    return best[1], m, S, keep, m2, S2 * c_alpha(p, 0.975)

rng = np.random.RandomState(3)
X = np.vstack([rng.multivariate_normal(np.zeros(3), np.eye(3) + 0.5, 80),
               rng.normal(5, 1, (20, 3))])  # 80 good, 20 outliers
H, m, S, keep, m2, S2 = my_mcd(X, seed=0)
ref = MinCovDet(random_state=0).fit(X)
assert set(H) == set(np.where(ref.raw_support_)[0])
assert np.allclose(m, ref.raw_location_)
assert np.allclose(S, ref.raw_covariance_)
assert (keep == ref.support_).all()
assert np.allclose(m2, ref.location_)
assert np.allclose(S2, ref.covariance_)
print("raw support", len(H), " reweighted support", keep.sum())
print("largest difference in covariance_:",
      np.abs(S2 - ref.covariance_).max())
