# Numbers quoted in the model answers of Chapter 8
import math
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score, train_test_split
np.set_printoptions(precision=4, suppress=True)

d = load_diabetes(); X, y = d.data, d.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
cols3 = [d.feature_names.index(n) for n in ("bmi", "bp", "s5")]

print("== Q1  two models whose means differ by 0.01 (training part, 331 patients, 5 folds)")
cv = KFold(5)
a = cross_val_score(LinearRegression(), Xtr, ytr, cv=cv)
b = cross_val_score(LinearRegression(), Xtr[:, cols3], ytr, cv=cv)
print("all ten  :", a.round(3), " mean", a.mean().round(4), " sd", a.std(ddof=1).round(4))
print("bmi,bp,s5:", b.round(3), " mean", b.mean().round(4), " sd", b.std(ddof=1).round(4))
dif = a - b
print("difference per fold:", dif.round(4), " mean", dif.mean().round(4), " sd", dif.std(ddof=1).round(4))
print("standard error of the mean score: %.4f ; of the mean difference: %.4f" % (a.std(ddof=1) / math.sqrt(5), dif.std(ddof=1) / math.sqrt(5)))
print("folds in which all ten is better:", int((dif > 0).sum()), "of 5")
means = []
for s in range(10):
    m_ = cross_val_score(LinearRegression(), Xtr, ytr, cv=KFold(5, shuffle=True, random_state=s)).mean(); means.append(m_)
print("mean score of all ten for 10 shuffles:", np.round(means, 3), " range", round(max(means) - min(means), 3))
rk = RepeatedKFold(n_splits=5, n_repeats=20, random_state=0)
ra = cross_val_score(LinearRegression(), Xtr, ytr, cv=rk); rb = cross_val_score(LinearRegression(), Xtr[:, cols3], ytr, cv=rk)
rd = (ra - rb).reshape(20, 5).mean(1)
print("20 x 5-fold, paired: mean difference %.4f ; per repeat min %.4f max %.4f ; better in %d of 100 folds" % ((ra - rb).mean(), rd.min(), rd.max(), int(((ra - rb) > 0).sum())))

print("\n== Q2  a courier: errors of 2 minutes and one wild record")
e = np.array([2., -2] * 50)
print("100 errors of size 2: MAE %.3f RMSE %.3f MSE %.3f" % (np.abs(e).mean(), np.sqrt((e ** 2).mean()), (e ** 2).mean()))
w = e.copy(); w[0] = 100.
print("one error of 100:    MAE %.3f RMSE %.3f MSE %.3f" % (np.abs(w).mean(), np.sqrt((w ** 2).mean()), (w ** 2).mean()))
print("share of the squared error due to that one record: %.3f ; of the absolute error: %.3f" % (100 ** 2 / (w ** 2).sum(), 100 / np.abs(w).sum()))

print("\n== Q3  rare disease, 99 per cent accuracy")
n, pos = 10000, 100
for name, (TP, FP) in {"flags nobody": (0, 0), "finds half, 50 false alarms": (50, 50), "finds all, 100 false alarms": (100, 100), "finds 90, 10 false alarms": (90, 10)}.items():
    FN = pos - TP; TN = n - pos - FP
    acc = (TP + TN) / n; rec = TP / pos; prec = TP / (TP + FP) if TP + FP else float("nan")
    print(f"{name:28s} TP {TP:3d} FP {FP:3d} FN {FN:3d} TN {TN:4d}  accuracy {acc:.4f}  recall {rec:.2f}  precision {prec:.2f}")

print("\n== Q4  the mean baseline and a negative R2")
m_tr = ytr.mean(); n_te = len(yte); sst = ((yte - yte.mean()) ** 2).sum(); sse = ((yte - m_tr) ** 2).sum()
print("training mean %.4f, mean of the test targets %.4f, n_test %d" % (m_tr, yte.mean(), n_te))
print("SST %.1f  SSE %.1f  SSE - SST = %.1f ; n (ybar - c)^2 = %.1f" % (sst, sse, sse - sst, n_te * (yte.mean() - m_tr) ** 2))
print("R2 = %.5f ; -n(ybar - c)^2 / SST = %.5f ; sklearn %.5f" % (1 - sse / sst, -n_te * (yte.mean() - m_tr) ** 2 / sst, r2_score(yte, np.full(n_te, m_tr))))
tr1, te1 = next(iter(KFold(5).split(X)))
c1 = y[tr1].mean(); y1 = y[te1]; s1 = ((y1 - y1.mean()) ** 2).sum(); e1 = ((y1 - c1) ** 2).sum()
print("fold 1: training mean %.2f, test mean %.2f, n %d, R2 = %.4f ; formula %.4f" % (c1, y1.mean(), len(y1), 1 - e1 / s1, -len(y1) * (y1.mean() - c1) ** 2 / s1))
print("DummyRegressor in the fold:", round(DummyRegressor().fit(X[tr1], y[tr1]).score(X[te1], y[te1]), 4))
