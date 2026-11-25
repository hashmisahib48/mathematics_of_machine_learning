"""Answers to the three projects of Chapter 3, computed with our class and with scikit-learn."""
import sys; sys.path.insert(0, '../code')
import numpy as np, statsmodels.api as sm
from linreg import LinearRegression as Ours
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as split
r = lambda v, k=4: round(float(v), k)
# ---------------------------------------------------------------- 3-1 stack loss (Brownlee 1965)
d = sm.datasets.stackloss.load_pandas().data; d.to_csv('../data/stackloss.csv', index=False)
y = d.STACKLOSS.values; X = d[['AIRFLOW', 'WATERTEMP', 'ACIDCONC']].values; x = X[:, 0]
n = len(y); xt = x - x.mean(); yt = y - y.mean()
Sxx, Sxy, Syy = (xt ** 2).sum(), (xt * yt).sum(), (yt ** 2).sum()
w = Sxy / Sxx; b = y.mean() - w * x.mean(); sse1 = ((y - w * x - b) ** 2).sum()
print('P1a n', n, 'xbar', r(x.mean()), 'ybar', r(y.mean()), 'Sxx', r(Sxx), 'Sxy', r(Sxy), 'Syy', r(Syy))
print('P1a w', r(w), 'b', r(b), 'R2', r(1 - sse1 / Syy))
for j, nm in enumerate(['air', 'water', 'acid']):
    print('P1 corr', nm, r(np.corrcoef(X[:, j], y)[0, 1], 3))
m = Ours().fit(X.tolist(), y.tolist()); e = y - np.array(m.predict(X.tolist()))
print('P1b w', [r(v) for v in m.coef_], 'b', r(m.intercept_), 'R2', r(m.score(X.tolist(), y.tolist())))
sk = LinearRegression().fit(X, y); print('P1b sk', [r(v) for v in sk.coef_], r(sk.intercept_))
k = np.argsort(-np.abs(e))[:3]; print('P1c largest errors days', (k + 1).tolist(), [r(e[i], 3) for i in k], 'rmse', r(np.sqrt((e ** 2).mean()), 3))
print('P1c sum e', r(e.sum(), 10), 'sum xe', [r((X[:, j] - X[:, j].mean()) @ e, 8) for j in range(3)])
print('P1d pred', r(m.predict([[60, 20, 85]])[0], 3), 'one-feature pred', r(w * 60 + b, 3))
keep = np.arange(n) != 20; m21 = Ours().fit(X[keep].tolist(), y[keep].tolist()); print('P1e without day 21', [r(v) for v in m21.coef_], r(m21.intercept_), 'R2', r(m21.score(X[keep].tolist(), y[keep].tolist())))
# ---------------------------------------------------------------- 3-2 Longley (1967)
d = sm.datasets.longley.load_pandas().data; d.to_csv('../data/longley.csv', index=False)
cols = ['GNPDEFL', 'GNP', 'UNEMP', 'ARMED', 'POP', 'YEAR']; X = d[cols].values; y = d.TOTEMP.values
Xc = X - X.mean(0); A = Xc.T @ Xc
print('P2a cond A', '%.3e' % np.linalg.cond(A), 'cond Xc', '%.3e' % np.linalg.cond(Xc))
C = np.corrcoef(X.T); print('P2a corr GNP-YEAR', r(C[1, 5], 4), 'GNP-POP', r(C[1, 4], 4), 'POP-YEAR', r(C[4, 5], 4))
m = Ours().fit(X.tolist(), y.tolist()); sk = LinearRegression().fit(X, y)
print('P2b ours', ['%.6g' % v for v in m.coef_], '%.8g' % m.intercept_)
print('P2b skl ', ['%.6g' % v for v in sk.coef_], '%.8g' % sk.intercept_, 'R2', r(sk.score(X, y), 6))
print('P2b max rel diff', '%.2e' % max(abs(a - b_) / abs(b_) for a, b_ in zip(m.coef_, sk.coef_)))
sub = ['GNP', 'UNEMP', 'ARMED']; Xs = d[sub].values; ms = LinearRegression().fit(Xs, y)
print('P2c three features', ['%.5g' % v for v in ms.coef_], '%.7g' % ms.intercept_, 'R2', r(ms.score(Xs, y), 4), 'cond', '%.3e' % np.linalg.cond((Xs - Xs.mean(0)).T @ (Xs - Xs.mean(0))))
tr = d.YEAR < 1962; a6 = LinearRegression().fit(X[tr], y[tr]); a3 = LinearRegression().fit(Xs[tr], y[tr])
print('P2d 1962 actual', y[~tr][0], 'six', r(a6.predict(X[~tr])[0], 1), 'three', r(a3.predict(Xs[~tr])[0], 1))
print('P2d six weights on 1947-61', ['%.4g' % v for v in a6.coef_])
# ---------------------------------------------------------------- 3-3 Engel (1857)
d = sm.datasets.engel.load_pandas().data; d.to_csv('../data/engel.csv', index=False)
x = d.income.values; y = d.foodexp.values
print('P3 n', len(x), 'income range', r(x.min(), 1), r(x.max(), 1), 'food range', r(y.min(), 1), r(y.max(), 1))
m = LinearRegression().fit(x[:, None], y); e = y - m.predict(x[:, None])
print('P3a w', r(m.coef_[0]), 'b', r(m.intercept_, 3), 'R2', r(m.score(x[:, None], y)))
lo = x < np.median(x); print('P3b error sd low half', r(e[lo].std(), 2), 'high half', r(e[~lo].std(), 2))
lx, ly = np.log(x), np.log(y); ml = LinearRegression().fit(lx[:, None], ly); el = ly - ml.predict(lx[:, None])
print('P3c elasticity', r(ml.coef_[0]), 'b', r(ml.intercept_), 'R2', r(ml.score(lx[:, None], ly)), 'sd low', r(el[lo].std(), 4), 'high', r(el[~lo].std(), 4))
for inc in [500, 1000, 2000]:
    f = np.exp(ml.predict([[np.log(inc)]])[0]); print('P3d share at', inc, r(f / inc, 3), 'food', r(f, 1))
xtr, xte, ytr, yte = split(x[:, None], y, test_size=0.25, random_state=0)
a = LinearRegression().fit(xtr, ytr); bl = LinearRegression().fit(np.log(xtr), np.log(ytr))
pl = np.exp(bl.predict(np.log(xte)))
from sklearn.metrics import r2_score, mean_absolute_error as mae
print('P3e test R2 linear', r(r2_score(yte, a.predict(xte))), 'log', r(r2_score(yte, pl)), 'MAE', r(mae(yte, a.predict(xte)), 1), r(mae(yte, pl), 1))
# ---- added parts: 3-1 (g) inverse use of the line, 3-3 (f) elasticity in use
d = sm.datasets.stackloss.load_pandas().data; x = d.AIRFLOW.values; y = d.STACKLOSS.values
w, b = np.polyfit(x, y, 1); print('P1g air flow for loss 10', r((10 - b) / w, 2))
d = sm.datasets.engel.load_pandas().data; el = np.polyfit(np.log(d.income), np.log(d.foodexp), 1)[0]
print('P3f rise for 10 % income', r(100 * (1.1 ** el - 1), 2), '%')
# ---------------------------------------------------------------- 3-4 Linnerud (Tenenhaus 1998), bundled with scikit-learn
from sklearn.datasets import load_linnerud
L = load_linnerud(as_frame=True); P = L.data.join(L.target); P.to_csv('../data/linnerud.csv', index=False)
y = L.data['Situps'].values; F = L.target[['Weight', 'Waist', 'Pulse']].values
print('P4 data', P.shape, list(P.columns))
x = F[:, 1]; w, b = np.polyfit(x, y, 1); m1 = LinearRegression().fit(x[:, None], y)
print('P4a waist w', r(w, 3), 'b', r(b, 2), 'R2', r(m1.score(x[:, None], y), 4))
m3 = LinearRegression().fit(F, y); print('P4b w', [r(v, 3) for v in m3.coef_], 'b', r(m3.intercept_, 2), 'R2', r(m3.score(F, y), 4))
print('P4c corr weight-waist', r(np.corrcoef(F[:, 0], F[:, 1])[0, 1], 3), 'corr weight-situps', r(np.corrcoef(F[:, 0], y)[0, 1], 3))
m2 = LinearRegression().fit(F[:, :2], y); print('P4c two features', [r(v, 3) for v in m2.coef_], r(m2.intercept_, 2), 'R2', r(m2.score(F[:, :2], y), 4))
print('P4d pred', r(m3.predict([[180, 34, 60]])[0], 1), r(m1.predict([[34]])[0], 1))
from sklearn.model_selection import cross_val_predict, LeaveOneOut
for nm, Z in [('waist', F[:, 1:2]), ('three', F)]:
    pr = cross_val_predict(LinearRegression(), Z, y, cv=LeaveOneOut()); print('P4e LOO R2', nm, r(1 - ((y - pr) ** 2).sum() / ((y - y.mean()) ** 2).sum(), 3))
# ---- added parts: 3-1 (h), 3-2 (e), 3-3 (g), 3-4 (f), 3-4 (g)
d = sm.datasets.stackloss.load_pandas().data; X = d[['AIRFLOW', 'WATERTEMP', 'ACIDCONC']].values
C = np.corrcoef(X.T); print('P1h corr air-water', r(C[0, 1], 3), 'air-acid', r(C[0, 2], 3), 'water-acid', r(C[1, 2], 3))
d = sm.datasets.longley.load_pandas().data; X = d[['GNPDEFL', 'GNP', 'UNEMP', 'ARMED', 'POP', 'YEAR']].values
Z = (X - X.mean(0)) / X.std(0); print('P2e cond A standardized', '%.3e' % np.linalg.cond(Z.T @ Z), 'cond A raw', '%.3e' % np.linalg.cond((X - X.mean(0)).T @ (X - X.mean(0))))
d = sm.datasets.engel.load_pandas().data; x = np.log(d.income.values); y = np.log(d.foodexp.values)
ml = LinearRegression().fit(x[:, None], y); el = y - ml.predict(x[:, None]); lo = x < np.median(x)
print('P3g sd of ln errors low', r(el[lo].std(), 4), 'high', r(el[~lo].std(), 4), 'corr e vs ln income', r(np.corrcoef(el, x)[0, 1], 6))
L = load_linnerud(as_frame=True); y = L.data['Situps'].values; F = L.target[['Weight', 'Waist', 'Pulse']].values
Zs = (F - F.mean(0)) / F.std(0); ms = LinearRegression().fit(Zs, y)
print('P4f standardized weights', [r(v, 2) for v in ms.coef_], 'b', r(ms.intercept_, 2), 'R2', r(ms.score(Zs, y), 4))
m2 = LinearRegression().fit(F[:, :2], y); print('P4g two-feature pred 180 lb 34 in', r(m2.predict([[180, 34]])[0], 1))
# ---- added parts: 3-3 (h), 3-4 (h) to (k)
d = sm.datasets.engel.load_pandas().data.sort_values('income'); sh = (d.foodexp / d.income).values
print('P3h mean food share poorest ten', r(sh[:10].mean(), 3), 'richest ten', r(sh[-10:].mean(), 3))
m3 = LinearRegression().fit(F, y); e3 = y - m3.predict(F); k = np.argsort(-np.abs(e3))[:3]
print('P4h largest errors men', (k + 1).tolist(), [r(e3[i], 1) for i in k], 'rmse', r(np.sqrt((e3 ** 2).mean()), 2))
p = F[:, 2]; mp = LinearRegression().fit(p[:, None], y); print('P4i pulse alone corr', r(np.corrcoef(p, y)[0, 1], 3), 'R2', r(mp.score(p[:, None], y), 4), 'w', r(mp.coef_[0], 3))
cm = F[:, 1] * 2.54; mc = LinearRegression().fit(cm[:, None], y); print('P4j waist in cm w', r(mc.coef_[0], 4), 'b', r(mc.intercept_, 2), 'R2', r(mc.score(cm[:, None], y), 4), 'ratio', r(mc.coef_[0] / m1.coef_[0], 4))
x5 = F[:5, 1]; y5 = y[:5]; xt = x5 - x5.mean(); yt5 = y5 - y5.mean()
print('P4k men 1-5 xbar', r(x5.mean(), 2), 'ybar', r(y5.mean(), 2), 'Sxx', r((xt ** 2).sum(), 3), 'Sxy', r((xt * yt5).sum(), 3), 'w', r((xt * yt5).sum() / (xt ** 2).sum(), 4), 'b', r(y5.mean() - (xt * yt5).sum() / (xt ** 2).sum() * x5.mean(), 3))
