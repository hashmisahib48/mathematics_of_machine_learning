"""Answers to the problems of Chapter 1 (for the solution manual), and
the data quoted in the statements of the problems."""
import numpy as np
from sklearn.datasets import load_wine, load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split as split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeRegressor
# ---- 1-1 wines
W = load_wine(); X, y = W.data, W.target; nm = list(W.feature_names)
print('1-1 shape', X.shape, 'counts', np.bincount(y))
a, b, c, d = split(X, y, test_size=0.25, random_state=0, stratify=y)
print('1-1 test counts', np.bincount(d), 'baseline',
      round(DummyClassifier().fit(a, c).score(b, d), 3))
for k in (1, 5, 15):
    m = KNeighborsClassifier(k).fit(a, c)
    print('1-1 k', k, round(m.score(b, d), 3), 'train', round(m.score(a, c), 3))
P = X[:, nm.index('proline')]; C = X[:, nm.index('color_intensity')]
H = X[:, nm.index('hue')]
print('1-1 ranges proline', P.min(), P.max(), 'color', C.min(), C.max(),
      'hue', H.min(), H.max())
rows, q = [127, 146, 37, 41, 141, 112], 105
print('1-1 data (row, proline, color, class):')
for i in rows + [q]: print('   ', i, P[i], C[i], y[i])
rP, rC = P.max() - P.min(), C.max() - C.min()
for tag, f in (('raw', (1, 1)), ('scaled', (rP, rC))):
    dd = np.hypot((P[rows] - P[q]) / f[0], (C[rows] - C[q]) / f[1])
    o = np.argsort(dd)
    print('1-1 hand', tag, 'dist', dd.round(3 if tag == 'scaled' else 1),
          '1-NN class', y[rows][o[0]], '3-NN classes', y[rows][o[:3]])
sc = MinMaxScaler().fit(a)
m = KNeighborsClassifier(5).fit(sc.transform(a), c)
print('1-1 k=5 scaled', round(m.score(sc.transform(b), d), 3))
# ---- 1-2 breast cancer
X, y = load_breast_cancer(return_X_y=True)
print('1-2 shape', X.shape, 'counts', np.bincount(y),
      'always benign', round(357 / 569, 3), 'always malignant', round(212 / 569, 3))
a, b, c, d = split(X, y, test_size=0.25, random_state=0, stratify=y)
print('1-2 test counts', np.bincount(d), 'baseline',
      round(DummyClassifier(strategy='most_frequent').fit(a, c).score(b, d), 3))
k5 = KNeighborsClassifier(5).fit(a, c); cm = confusion_matrix(d, k5.predict(b))
print('1-2 5-NN confusion (rows true malignant, benign)', cm.tolist(),
      'acc', round(k5.score(b, d), 3), 'found malignant', round(cm[0, 0] / cm[0].sum(), 3),
      'found benign', round(cm[1, 1] / cm[1].sum(), 3))
m = LogisticRegression(max_iter=10000).fit(a, c); p = m.predict(b)
print('1-2 logistic', round(m.score(b, d), 3), 'malignant called benign',
      int(((d == 0) & (p == 1)).sum()), 'benign called malignant', int(((d == 1) & (p == 0)).sum()))
from sklearn.preprocessing import StandardScaler
sc2 = StandardScaler().fit(a)
k5s = KNeighborsClassifier(5).fit(sc2.transform(a), c)
print('1-2 5-NN standardized', round(k5s.score(sc2.transform(b), d), 3))
# ---- 1-3 diabetes
D = load_diabetes(); X, y = D.data, D.target
a, b, c, d = split(X, y, test_size=0.25, random_state=0)
t1 = DecisionTreeRegressor(max_depth=1, random_state=0).fit(a, c)
j, thr = t1.tree_.feature[0], t1.tree_.threshold[0]
lo, hi = t1.tree_.value[1, 0, 0], t1.tree_.value[2, 0, 0]
print('1-3 stump feature', D.feature_names[j], 'threshold', round(thr, 4),
      'leaf means', round(lo, 1), round(hi, 1), 'training mean', round(c.mean(), 1))
print('1-3 six test patients (s5, progression, stump prediction, error):')
for i in range(6):
    pr = lo if b[i, j] <= thr else hi
    print('   ', round(b[i, j], 4), d[i], round(pr, 1), round(d[i] - pr, 1))
e1 = np.array([d[i] - (lo if b[i, j] <= thr else hi) for i in range(6)])
e0 = d[:6] - c.mean()
print('1-3 hand MSE stump', round((e1 ** 2).mean(), 0), 'RMSE', round(np.sqrt((e1 ** 2).mean()), 1),
      'constant MSE', round((e0 ** 2).mean(), 0), 'RMSE', round(np.sqrt((e0 ** 2).mean()), 1),
      'R2 six', round(1 - (e1 ** 2).sum() / ((d[:6] - d[:6].mean()) ** 2).sum(), 3))
print('1-3 stump test R2', round(t1.score(b, d), 3))
for k in (1, 2, 3, 4, 6, 8, 12):
    t = DecisionTreeRegressor(max_depth=k, random_state=0).fit(a, c)
    print('1-3 depth', k, 'train', round(t.score(a, c), 3), 'test', round(t.score(b, d), 3))
print('1-3 linear test R2', round(LinearRegression().fit(a, c).score(b, d), 3))
a1, b1, c1, d1 = split(X, y, test_size=0.25, random_state=1)
for k in (1, 2, 3, 4, 6, 8, 12):
    t = DecisionTreeRegressor(max_depth=k, random_state=0).fit(a1, c1)
    print('1-3 seed 1 depth', k, 'train', round(t.score(a1, c1), 3), 'test', round(t.score(b1, d1), 3))
