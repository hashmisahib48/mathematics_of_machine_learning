# Hand checks for Problems 8A.1 - 8A.4 (every number quoted in the solutions)
import math
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, root_mean_squared_error)
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score, train_test_split
np.set_printoptions(precision=4, suppress=True)

print("== 8A.1  y = 2, 5, 7, 10; predictions 3, 4, 7, 12")
y = np.array([2., 5, 7, 10]); yh = np.array([3., 4, 7, 12]); r = y - yh; m = y.mean()
print("residuals:", r, " |r|:", np.abs(r), " r^2:", r ** 2, " mean of y:", m, " (y-ybar)^2:", (y - m) ** 2)
sse, sst, sab = (r ** 2).sum(), ((y - m) ** 2).sum(), np.abs(r).sum()
print("SSE =", sse, " SST =", sst, " sum|r| =", sab)
mse = sse / 4; print("MSE =", mse, " RMSE =", math.sqrt(mse), " MAE =", sab / 4)
print("R2 =", 1 - sse / sst, " second form: 1 - MSE/(SST/n) =", 1 - mse / (sst / 4))
print("sklearn:", mean_squared_error(y, yh), root_mean_squared_error(y, yh), mean_absolute_error(y, yh), r2_score(y, yh))
sst_w = ((y - yh.mean()) ** 2).sum(); print("slip: SST about the mean of the predictions", yh.mean(), "=", sst_w, " R2 =", round(1 - sse / sst_w, 4))
print("RMSE >= MAE:", math.sqrt(mse) >= sab / 4)
print("share of SSE from the two larger residuals:", (r[0] ** 2 + r[3] ** 2) / sse, " of SAE:", (abs(r[0]) + abs(r[3])) / sab)

print("\n== 8A.2  1000 patients, 50 diseased, 60 flagged, 40 of the flagged diseased")
TP, FP = 40, 60 - 40; FN = 50 - TP; TN = 1000 - TP - FP - FN
print("TP FP FN TN:", TP, FP, FN, TN, " sum", TP + FP + FN + TN)
print("accuracy", (TP + TN) / 1000, " precision", TP / (TP + FP), " recall", TP / (TP + FN))
p, rc = TP / (TP + FP), TP / (TP + FN)
print("F1 =", 2 * p * rc / (p + rc), " = 2TP/(2TP+FP+FN) =", 2 * TP / (2 * TP + FP + FN))
yt = np.r_[np.ones(40), np.ones(10), np.zeros(20), np.zeros(930)]; yp = np.r_[np.ones(40), np.zeros(10), np.ones(20), np.zeros(930)]
print("sklearn:", confusion_matrix(yt, yp).tolist(), precision_score(yt, yp), recall_score(yt, yp), round(f1_score(yt, yp), 4))
yn = np.zeros(1000)
print("flags nobody: accuracy", (yn == yt).mean(), " recall", recall_score(yt, yn), " precision (0/0):", precision_score(yt, yn, zero_division=0), " TN =", 950, " FN =", 50)

print("\n== 8A.3  leave one out for 2, 4, 9")
v = np.array([2., 4, 9]); e = []
for k in range(3):
    mu = np.delete(v, k).mean(); e.append(v[k] - mu); print(f"hold out {v[k]:.0f}: training mean {mu}, error {v[k] - mu}, square {(v[k] - mu) ** 2}")
e = np.array(e); loo = (e ** 2).mean(); tr = ((v - v.mean()) ** 2).mean()
print("sum of squares", (e ** 2).sum(), " LOO MSE =", loo, " mean", v.mean(), " squares", (v - v.mean()) ** 2, " training MSE =", tr, " ratio", loo / tr)
print("factor (n/(n-1))^2 for n = 10:", round((10 / 9) ** 2, 4))
print("error = n/(n-1) * (y_i - ybar):", 1.5 * (v - v.mean()), "  (n/(n-1))^2 =", (3 / 2) ** 2)
s = cross_val_score(DummyRegressor(), v[:, None], v, cv=LeaveOneOut(), scoring="neg_mean_squared_error")
print("sklearn LeaveOneOut:", -s.mean(), " per sample", -s)

print("\n== 8A.4  counting samples")
n_te = math.ceil(0.2 * 150); print("test:", 0.2 * 150, "->", n_te, " train:", 150 - n_te, " per class in test:", n_te // 3, " in train:", (150 - n_te) // 3)
yy = np.repeat([0, 1, 2], 50)
_, _, ytr, yte = train_test_split(np.zeros((150, 1)), yy, test_size=0.2, random_state=0, stratify=yy)
print("sklearn: test", len(yte), np.bincount(yte).tolist(), " train", len(ytr), np.bincount(ytr).tolist())
tr, te = next(iter(KFold(5).split(np.zeros(120)))); print("KFold(5) on 120: fit uses", len(tr), " scores", len(te))
print("103 = 5 *", 103 // 5, "+", 103 % 5, " fold sizes:", [len(t) for _, t in KFold(5).split(np.zeros(103))])
