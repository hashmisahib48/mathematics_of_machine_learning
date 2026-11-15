# Problem 10C.2: a leave-one-out curve whose minimum sits at the edge of the grid
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_score, train_test_split
x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]); y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])
def loo(X, y, a):
    Xc, yc = X - X.mean(0), y - y.mean()
    S = np.linalg.inv(Xc.T @ Xc + a * np.eye(X.shape[1])); h = 1 / len(y) + np.einsum("ij,jk,ik->i", Xc, S, Xc)
    r = yc - Xc @ (S @ Xc.T @ yc); return np.mean((r / (1 - h)) ** 2)
print("delivery data, one feature: leave-one-out error against alpha")
for a in (0, 1e-3, 0.01, 0.1, 0.3, 1, 3, 10):
    print(f"  alpha = {a:7.3f}   LOO = {loo(x, y, a):.4f}")
cv = -cross_val_score(LinearRegression(), x, y, cv=LeaveOneOut(), scoring="neg_mean_squared_error").mean()
print("LinearRegression with LeaveOneOut:", round(cv, 4))
r = RidgeCV(alphas=np.logspace(-6, 1, 8), store_cv_results=True).fit(x, y)
print("RidgeCV on a grid 1e-6 ... 10: alpha_ =", r.alpha_, " mean cv_results_:", r.cv_results_.mean(0).round(4))
print("condition number of Xc'Xc (one column): 1, leverages at alpha = 0:", np.round(1 / 5 + (x[:, 0] - 3) ** 2 / 10, 3))
print("slope: least squares", np.polyfit(x[:, 0], y, 1)[0], " ridge alpha = 0.1:", round(float(Ridge(alpha=0.1).fit(x, y).coef_[0]), 4))

print("\ndiabetes training part (Section 10.12): leave-one-out curve of RidgeCV")
d = load_diabetes(); Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=0.25, random_state=0)
al = np.logspace(-3, 1, 41); m = RidgeCV(alphas=al, store_cv_results=True).fit(Xtr, ytr)
mse = m.cv_results_.mean(0); k = int(np.argmin(mse))
print(f"minimum at alpha = {al[k]:.4f}, index {k} of {len(al) - 1};  LOO MSE there {mse[k]:.1f}")
for a in (0.001, 0.01, 0.0501, 0.3, 1, 10):
    i = int(np.argmin(np.abs(al - a))); print(f"  alpha = {al[i]:7.4f}  LOO MSE = {mse[i]:.1f}")
