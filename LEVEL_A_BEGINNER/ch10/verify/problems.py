import sys, math, numpy as np, pandas as pd
sys.path.insert(0, '../code')
from kmeans import KMeans as KM, sqdist
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine, load_digits, load_sample_image
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
# 10-1
W = load_wine(); X, y = W.data, W.target
for nm, Z in [('raw', X), ('std', StandardScaler().fit_transform(X))]:
    lab = KMeans(3, n_init=10, random_state=0).fit_predict(Z)
    print('10-1', nm, 'ARI', round(adjusted_rand_score(y, lab), 4)); print(pd.crosstab(y, lab))
# 10-2
img = load_sample_image('flower.jpg') / 255.0; P = img.reshape(-1, 3); n = len(P)
rng = np.random.default_rng(0); S = P[rng.choice(n, 2000, replace=False)]
for k in (2, 4, 8, 16, 32, 64):
    m = KMeans(k, n_init=4, random_state=0).fit(S); Q = m.cluster_centers_[m.predict(P)]
    bits = 24 * k + n * math.log2(k)
    print('10-2 k', k, 'rms', round(float(np.sqrt(np.mean((P - Q) ** 2))), 4), 'size ratio', round(bits / (24 * n), 4))
# 10-3
D = load_digits(); lab = KMeans(10, n_init=10, random_state=0).fit_predict(D.data)
mp = {c: np.bincount(D.target[lab == c]).argmax() for c in range(10)}
pred = np.array([mp[c] for c in lab]); print('10-3 map', mp, 'acc', round((pred == D.target).mean(), 4), 'ARI', round(adjusted_rand_score(D.target, lab), 4))
# numerical parts folded into the case studies
img = load_sample_image('flower.jpg') / 255.0; P = img.reshape(-1, 3); rng = np.random.default_rng(0); S = P[rng.choice(len(P), 2000, replace=False)]
m4 = KMeans(4, n_init=4, random_state=0).fit(S); print('10-2c palette', m4.cluster_centers_.round(3).tolist(), 'pixel0', P[0].round(3), 'sqd', ((m4.cluster_centers_.round(3) - P[0].round(3)) ** 2).sum(1).round(4), 'nearest', m4.predict(P[[0]]))
import statsmodels.api as sm
xa = sm.datasets.stackloss.load_pandas().data.AIRFLOW.values
for init in ([[50], [80]], [[50], [56]]):
    m = KM(2, init=init, max_iter=100).fit([[v] for v in xa]); print('10-4ab init', init, 'centers', [round(c_[0], 4) for c_ in m.cluster_centers_], 'J', round(m.inertia_, 4), 'iters', m.n_iter_)
Dp = (xa - 80) ** 2; print('10-4c', {int(v): round(float(Dp[xa == v].sum() / Dp.sum()), 4) for v in sorted(set(xa))})
for k in (2, 3, 4):
    m = KMeans(k, n_init=10, random_state=0).fit(xa[:, None]); print('10-4d k', k, np.sort(m.cluster_centers_.ravel()).round(3), round(m.inertia_, 3))
print('10-3c ops', 1797 * 10 * 64)
print('10-3c 10 starts x 20 iters', 1797 * 10 * 64 * 200)
# 10-4(d): the K = 4 result depends on the start; exhaustive search over the partitions of the sorted days
print('10-4d K=4 init=random n_init=1 by random_state', [round(KMeans(4, init='random', n_init=1, random_state=s).fit(xa[:, None]).inertia_, 3) for s in range(10)])
from itertools import combinations
xs = np.sort(xa.astype(float)); best = min(sum(((g - g.mean()) ** 2).sum() for g in np.split(xs, c)) for c in combinations(range(1, len(xs)), 3))
print('10-4d K=4 global minimum by exhaustive search', round(float(best), 3))
