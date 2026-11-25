"""Answers to the problems of Chapter 3, computed."""
import sys; sys.path.insert(0, '../code')
import numpy as np
from linreg import LinearRegression as Ours
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split as split
x = [10, 20, 30, 40, 50]; y = [2.1, 2.9, 4.2, 4.8, 6.1]
m = Ours().fit([[v] for v in x], y)
e = [a - b for a, b in zip(y, m.predict([[v] for v in x]))]
print('3-1 w', round(m.coef_[0], 4), 'b', round(m.intercept_, 4), 'pred35', round(m.predict([[35]])[0], 3))
print('3-2 e', [round(v, 3) for v in e], 'sum', round(sum(e), 12), 'sumxe', round(sum((a - 30) * v for a, v in zip(x, e)), 12), 'R2', round(m.score([[v] for v in x], y), 4))
A = np.array([[4., 2], [2, 3]]); c = np.array([10., 8]); print('3-4', np.linalg.solve(A, c))
A = np.array([[1., 3], [3, 10]]); c = np.array([5., 17]); print('3-5', np.linalg.solve(A, c), 'm after swap', 1 / 3, 'a22', 3 - 10 / 3, 'c2', 5 - 17 / 3)
X = [[2, 1], [4, 2], [5, 1], [7, 3], [9, 2]]; yy = [14, 23, 24, 35, 38]
m1 = Ours().fit(X, yy); m2 = Ours().fit([[r[0] * 1000, r[1]] for r in X], yy)
print('3-7 w1', round(m1.coef_[0], 6), round(m2.coef_[0], 9), 'b same', round(m1.intercept_, 6) == round(m2.intercept_, 6))
sw = np.array([1, 1, 2, 1, 1.]); sk = LinearRegression().fit(np.array(X, float), yy, sample_weight=sw)
print('3-8', np.round(sk.coef_, 4), round(sk.intercept_, 4))
D, t = load_diabetes(return_X_y=True, as_frame=True)
Xtr, Xte, ytr, yte = split(D[['bmi', 's5']], t, test_size=0.25, random_state=0)
print('3-9 R2', round(LinearRegression().fit(Xtr, ytr).score(Xte, yte), 3))
m3 = Ours(fit_intercept=False).fit([[1, r[0], r[1]] for r in X], yy)
print('3-10', [round(v, 4) for v in m3.coef_])
# ---- added numerical problems 3-11 to 3-18
x = [1, 2, 3, 4, 5, 6]; y = [2.8, 4.1, 6.2, 7.9, 10.1, 11.8]
m = Ours().fit([[v] for v in x], y); print('3-11 w', round(m.coef_[0], 4), 'b', round(m.intercept_, 4), 'R2', round(m.score([[v] for v in x], y), 4), 'pred8', round(m.predict([[8]])[0], 3))
X = [[1, 2], [2, 1], [3, 4], [4, 3]]; yy = [6, 5, 11, 10]
mm = Ours().fit(X, yy); Xc = np.array(X, float) - np.mean(X, 0); yc = np.array(yy, float) - np.mean(yy)
print('3-12 A', (Xc.T @ Xc).tolist(), 'c', (Xc.T @ yc).tolist(), 'w', [round(v, 4) for v in mm.coef_], 'b', round(mm.intercept_, 4))
yt = np.array([3., 5, 7, 9]); yp = np.array([3.4, 4.6, 7.3, 8.7])
sst = ((yt - yt.mean()) ** 2).sum(); ssr = ((yt - yp) ** 2).sum(); print('3-13 SStot', sst, 'SSres', round(ssr, 4), 'R2', round(1 - ssr / sst, 4), 'RMSE', round(np.sqrt(ssr / 4), 4))
A = np.array([[2., 1, 1], [4, 3, 3], [8, 7, 9]]); c = np.array([4., 10, 24]); print('3-14 w', np.linalg.solve(A, c), 'first pivot 8, multipliers', 2 / 8, 4 / 8)
F = [0.0, 0.0]; ext = np.array([2., 4, 6, 8, 10]); force = np.array([0.9, 2.1, 2.9, 4.2, 5.0])
k = (ext @ force) / (ext @ ext); print('3-15 k', round(k, 4), 'N/mm with intercept', [round(v, 4) for v in np.polyfit(ext, force, 1)])
yo = [14, 23, 24, 35, 58]; mo = Ours().fit([[r[0]] for r in [[2], [4], [5], [7], [9]]], yo); m0 = Ours().fit([[2], [4], [5], [7], [9]], [14, 23, 24, 35, 38])
print('3-16 w before', round(m0.coef_[0], 4), 'after', round(mo.coef_[0], 4), 'b', round(m0.intercept_, 4), round(mo.intercept_, 4))
Dtr, Dte, ttr, tte = split(D[['bmi']], t, test_size=0.25, random_state=0); sb = LinearRegression().fit(Dtr, ttr)
print('3-17 bmi only R2', round(sb.score(Dte, tte), 3), 'w', round(sb.coef_[0], 1), 'b', round(sb.intercept_, 1))
xs = np.array([0., 1, 2, 3, 4]); ys = np.array([1.0, 2.7, 7.4, 20.1, 54.6])
ml = Ours().fit([[v] for v in xs], np.log(ys).tolist()); print('3-18 ln-fit w', round(ml.coef_[0], 4), 'b', round(ml.intercept_, 4), 'a=e^b', round(np.exp(ml.intercept_), 4))
# ---- 3-19 to 3-24
m = Ours().fit([[0], [1], [2], [3]], [1, 3, 4, 6]); print('3-19 w', round(m.coef_[0], 4), 'b', round(m.intercept_, 4))
yv = np.array([4., 7, 10]); print('3-20 b', yv.mean(), 'SSres', ((yv - yv.mean()) ** 2).sum())
x = [10, 20, 30, 40, 50]; y = [2.1, 2.9, 4.2, 4.8, 6.1]; xf = [v * 9 / 5 + 32 for v in x]
mf = Ours().fit([[v] for v in xf], y); print('3-21 w', round(mf.coef_[0], 5), 'b', round(mf.intercept_, 4), 'R2', round(mf.score([[v] for v in xf], y), 4))
Xn = np.array([[1, 1], [2, 2], [3, 3.001]]); Xc = Xn - Xn.mean(0); print('3-22 cond', '%.3g' % np.linalg.cond(Xc.T @ Xc))
x = [1, 2, 3, 4, 5, 6]; y = [2.8, 4.1, 6.2, 7.9, 10.1, 11.8]
mt = Ours().fit([[v] for v in x[:4]], y[:4]); p = mt.predict([[5], [6]]); e = np.array(y[4:]) - np.array(p)
print('3-23 w', round(mt.coef_[0], 4), 'b', round(mt.intercept_, 4), 'pred', [round(v, 3) for v in p], 'RMSE', round(float(np.sqrt((e ** 2).mean())), 4))
xq = [-2, -1, 0, 1, 2]; yq = [4.1, 1.2, 0.1, 0.9, 3.9]; mq = Ours().fit([[v, v * v] for v in xq], yq)
print('3-24 w', [round(v, 4) for v in mq.coef_], 'b', round(mq.intercept_, 4), 'R2', round(mq.score([[v, v * v] for v in xq], yq), 4))
# ---- 3-25 to 3-28
I = np.array([0.5, 1.0, 1.5, 2.0]); V = np.array([1.1, 2.0, 3.1, 3.9]); print('3-25 R', round((I @ V) / (I @ I), 4))
xg = np.array([1., 2, 3]); yg = np.array([2., 4, 7]); eg = yg - (1 * xg + 0); print('3-26 dJdw', -2 * (xg @ eg), 'dJdb', -2 * eg.sum())
X = [[2, 1], [4, 2], [5, 1], [7, 3], [9, 2]]; yy = [14, 23, 24, 35, 38]; m1 = Ours().fit(X, yy)
print('3-27', [round(v, 3) for v in m1.predict([[3, 4], [12, 1]])])
yt = np.array([1., 2, 3]); yp = np.array([3., 2, 1]); print('3-28 R2', 1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
# ---- 3-29 to 3-32
x = [1, 2, 3, 4, 5, 6]; y = [2.8, 4.1, 6.2, 7.9, 10.1, 11.8]; m = Ours().fit([[v] for v in x], y); e = np.array(y) - np.array(m.predict([[v] for v in x]))
print('3-29 e', [round(float(v), 3) for v in e], 'sum', round(float(e.sum()), 10), 'sumxe', round(float(np.array(x) @ e), 10))
X = [[2, 1], [4, 2], [5, 1], [7, 3], [9, 2]]; ys = [60 * v for v in [14, 23, 24, 35, 38]]; ms = Ours().fit(X, ys)
print('3-30 w', [round(float(v), 2) for v in ms.coef_], 'b', round(float(ms.intercept_), 2))
