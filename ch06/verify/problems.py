import numpy as np, statsmodels.api as sm
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict, LeaveOneOut, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
X, y = load_breast_cancer(return_X_y=True)
for k in (1, 3, 5, 11, 21):
    a = cross_val_score(KNeighborsClassifier(k), X, y, cv=5).mean(); b = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(k)), X, y, cv=5).mean()
    print('6-1 k', k, 'raw', round(a, 3), 'std', round(b, 3))
a, b, c, d = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
m = make_pipeline(StandardScaler(), KNeighborsClassifier(5)).fit(a, c); print('6-1 confusion\n', confusion_matrix(d, m.predict(b)))
X, y = load_digits(return_X_y=True); a, b, c, d = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
for k in (1, 3, 7):
    m = KNeighborsClassifier(k).fit(a, c); p = m.predict(b); print('6-2 k', k, 'acc', round((p == d).mean(), 4), 'errors', int((p != d).sum()))
m = KNeighborsClassifier(3).fit(a, c); p = m.predict(b); print('6-2 errors true->pred', [(int(t), int(q)) for t, q in zip(d, p) if t != q])
s = sm.datasets.stackloss.load_pandas().data; X = s[['AIRFLOW', 'WATERTEMP', 'ACIDCONC']].values; y = s.STACKLOSS.values
for name, m in [('linear', LinearRegression())] + [(f'knn{k}', make_pipeline(StandardScaler(), KNeighborsRegressor(k))) for k in (1, 2, 3, 5)]:
    p = cross_val_predict(m, X, y, cv=LeaveOneOut()); print('6-3', name, 'LOO RMSE', round(np.sqrt(((y - p) ** 2).mean()), 3))

# ---------------- numerical parts folded into the case studies
import math
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import statsmodels.api as sm
B = load_breast_cancer(); X, y = B.data, B.target; a, b, c, d = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
j1, j2 = list(B.feature_names).index('mean area'), list(B.feature_names).index('mean smoothness')
sc = StandardScaler().fit(a); mu, sd = sc.mean_[[j1, j2]], sc.scale_[[j1, j2]]
t0, t1 = b[0, [j1, j2]], b[1, [j1, j2]]
raw = (t0 - t1) ** 2; st = ((t0 - t1) / sd) ** 2
print('6-1d means', mu.round(5), 'sd', sd.round(5), 't0', t0, 't1', t1)
print('6-1d raw dist', round(math.sqrt(raw.sum()), 3), 'area share', round(raw[0] / raw.sum(), 6), 'std dist', round(math.sqrt(st.sum()), 4), 'area share', round(st[0] / st.sum(), 4))
Z = sc.transform(a); zb = sc.transform(b); m = KNeighborsClassifier(5).fit(Z, c)
dist, nb = m.kneighbors(zb[[13]]); w = 1 / dist[0]
print('6-1e tumor 13 dist', dist.round(3), 'classes', c[nb], 'uniform P', (c[nb[0]] == 1).mean(), 'dw P', round((w * (c[nb[0]] == 1)).sum() / w.sum(), 4), 'true', d[13])
print('6-2d train', 1797 - 450, 'ops per image', 3 * 1347 * 64, 'for 450', 3 * 1347 * 64 * 450)
s_ = sm.datasets.stackloss.load_pandas().data; xa = s_.AIRFLOW.values; ya = s_.STACKLOSS.values
dd = np.abs(xa - 72); o = np.argsort(dd, kind='stable')[:2]
print('6-3c neighbors', xa[o], ya[o], 'uniform', ya[o].mean(), 'dw', round((ya[o] / dd[o]).sum() / (1 / dd[o]).sum(), 3), 'line', round(-44.132 + 1.0203 * 72, 2))
# 6-4 iris by hand
from sklearn.datasets import load_iris
Xi, yi = load_iris(return_X_y=True); Pi = Xi[:, 2:]; rows = [52, 56, 63, 77, 106, 119, 126, 134]; q = Pi[83]
di = np.sqrt(((Pi[rows] - q) ** 2).sum(1)); o = np.argsort(di, kind='stable')
print('6-4 q', q, 'true', yi[83]); print('6-4 table', [(r, Pi[r].tolist(), yi[r], round(di[k], 4)) for k, r in enumerate(rows)])
for kk in (3, 5):
    N = o[:kk]; cls_ = yi[np.array(rows)[N]]; w = 1 / di[N]
    print('6-4 k', kk, 'votes ve', int((cls_ == 1).sum()), 'vi', int((cls_ == 2).sum()), 'weighted ve', round(w[cls_ == 1].sum(), 3), 'vi', round(w[cls_ == 2].sum(), 3))
mask = yi > 0
for kk in (1, 3, 5, 7, 11, 15):
    print('6-4c k', kk, round(cross_val_score(KNeighborsClassifier(kk), Pi[mask], yi[mask], cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean(), 4))
p8 = cross_val_predict(KNeighborsClassifier(3), Pi[rows], yi[rows], cv=LeaveOneOut()); print('6-4d LOO k3 pred', p8, 'true', yi[rows], 'errors', int((p8 != yi[rows]).sum()))
# ---------------- parts added in the revision
Bc = load_breast_cancer(); print('6-1f benign share', round(Bc.target.mean(), 4), 'of', len(Bc.target))
x90 = 90; dd = np.abs(xa - x90); o = np.argsort(dd, kind='stable')[:3]
print('6-3e air flow max', xa.max(), 'x=90 neighbors', xa[o], ya[o], 'knn3', ya[o].mean(), 'line', round(-44.132 + 1.0203 * 90, 2))
print('6-4e sd of petal length, width over 100 flowers', Pi[mask].std(0, ddof=0).round(4), 'ddof1', Pi[mask].std(0, ddof=1).round(4))
Z = StandardScaler().fit(Pi[mask]).transform(Pi[mask]); zq = (q - Pi[mask].mean(0)) / Pi[mask].std(0)
rr = np.array(rows) - 50; dz = np.sqrt(((Z[rr] - zq) ** 2).sum(1)); oz = np.argsort(dz, kind='stable')
print('6-4e standardized distances', dz.round(3), 'order', np.array(rows)[oz], 'k3 classes', yi[np.array(rows)[oz[:3]]])
