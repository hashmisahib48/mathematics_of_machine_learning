"""Answers to the problems of Chapter 2 (for the solution manual)."""
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
X = load_iris().data; print('2-1 means', X.mean(0).round(3), 'sd', X.std(0, ddof=1).round(3))
C = np.corrcoef(X.T); print('2-1 corr\n', C.round(3))
x, y = X[:, 2], X[:, 3]; w = (x - x.mean()) @ (y - y.mean()) / ((x - x.mean()) ** 2).sum(); print('2-1 slope', round(w, 4), 'b', round(y.mean() - w * x.mean(), 4))
B = load_breast_cancer(); r = B.data[:, 0] > 15; m = B.target == 0
pm = m.mean(); pr_m = r[m].mean(); pr_b = r[~m].mean(); pr = r.mean()
print('2-2 P(M)', round(pm, 4), 'P(R|M)', round(pr_m, 4), 'P(R|B)', round(pr_b, 4), 'P(R)', round(pr, 4), 'P(M|R)', round(pr_m * pm / pr, 4), 'direct', round(m[r].mean(), 4))
W = load_wine(); X = W.data; d = np.sqrt(((X - X[0]) ** 2).sum(1)); d[0] = np.inf; k = d.argmin()
print('2-3 raw nearest', k, 'class', W.target[k], 'dist', round(d[k], 3), 'range proline', X[:, 12].min(), X[:, 12].max(), 'hue', X[:, 10].min(), X[:, 10].max())
Z = (X - X.mean(0)) / X.std(0, ddof=1); d = np.sqrt(((Z - Z[0]) ** 2).sum(1)); d[0] = np.inf; k = d.argmin()
print('2-3 std nearest', k, 'class', W.target[k], 'dist', round(d[k], 3))
correct = lambda M: np.mean([W.target[np.argsort(np.sqrt(((M - M[i]) ** 2).sum(1)))[1]] == W.target[i] for i in range(len(M))])
print('2-3 leave-one-out 1-NN accuracy raw', round(correct(X), 3), 'standardized', round(correct(Z), 3))
from sklearn.datasets import load_diabetes
D = load_diabetes(); x = D.data[:, 2]; y = D.target; xt = x - x.mean(); yt = y - y.mean()
w = (xt @ yt) / (xt @ xt); print('2-4 w formula', round(w, 2), 'b', round(y.mean() - w * x.mean(), 2), 'Sxx', round(xt @ xt, 5))
for eta in (0.5, 1.0, 1.2):
    v = 0.0
    for k in range(50): v -= eta * (-2 * xt @ (yt - v * xt))
    print('2-4 eta', eta, 'w after 50', round(v, 2))
# ---- data quoted in the statements of the problems
X = load_iris().data; x, y = X[:, 2], X[:, 3]
sxx = ((x - x.mean()) ** 2).sum(); syy = ((y - y.mean()) ** 2).sum()
sxy = (x - x.mean()) @ (y - y.mean())
print('data 2-1 xbar', round(x.mean(), 3), 'ybar', round(y.mean(), 3),
      'Sxx', round(sxx, 2), 'Syy', round(syy, 2), 'Sxy', round(sxy, 2))
print('data 2-1 r', round(sxy / (sxx * syy) ** 0.5, 4),
      'slope', round(sxy / sxx, 4), 'b', round(y.mean() - sxy / sxx * x.mean(), 4))
B = load_breast_cancer(); r = B.data[:, 0] > 15; m = B.target == 0
print('data 2-2 n', len(r), 'malignant', m.sum(), 'benign', (~m).sum(),
      'R&M', (r & m).sum(), 'R&B', (r & ~m).sum(), 'R', r.sum())
W = load_wine(); X = W.data
for i in (0, 1):
    print('data 2-3 wine', i, 'proline', X[i, 12], 'hue', X[i, 10])
print('data 2-3 proline mean', round(X[:, 12].mean(), 2), 'sd',
      round(X[:, 12].std(ddof=1), 2), 'hue mean', round(X[:, 10].mean(), 3),
      'sd', round(X[:, 10].std(ddof=1), 3))
d = X[1, [12, 10]] - X[0, [12, 10]]
print('data 2-3 d2 raw', round(float(np.sqrt((d ** 2).sum())), 3),
      'proline share', round(float(d[0] ** 2 / (d ** 2).sum()), 6))
zs = np.array([X[:, 12].std(ddof=1), X[:, 10].std(ddof=1)])
dz = d / zs
print('data 2-3 std diffs', dz.round(3), 'd', round(float(np.sqrt((dz ** 2).sum())), 3))
D = load_diabetes(); x = D.data[:, 2]; y = D.target; xt = x - x.mean()
print('data 2-4 n', len(x), 'Sxx', round(xt @ xt, 4), 'Sxy',
      round(xt @ (y - y.mean()), 1), 'ybar', round(y.mean(), 2))
