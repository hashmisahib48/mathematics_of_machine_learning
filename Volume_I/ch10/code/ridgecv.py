# Leave-one-out errors of ridge: hat matrix, refits and RidgeCV.
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV

x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])
n = len(y)

def loo_hat(X, y, a):                      # Eq. 10.9-4, 10.9-5
    Xc, yc = X - X.mean(axis=0), y - y.mean()
    S = np.linalg.inv(Xc.T @ Xc + a * np.eye(X.shape[1]))
    h = 1 / n + np.einsum("ij,jk,ik->i", Xc, S, Xc)   # diag of H
    r = yc - Xc @ (S @ Xc.T @ yc)          # ordinary residuals
    return r, h, r / (1 - h)

r, h, e = loo_hat(x, y, 1.0)
print("alpha = 1")
print("h_ii =", np.round(h, 4))
print("r_i  =", np.round(r, 4))
print("e_i  =", np.round(e, 4))

alphas = [0.1, 1.0, 10.0]
cv = RidgeCV(alphas=alphas, store_cv_results=True).fit(x, y)
print("\nalpha   hat formula   n refits   RidgeCV")
for k, a in enumerate(alphas):
    refit = []
    for i in range(n):                     # n separate fits
        m = Ridge(alpha=a).fit(np.delete(x, i, 0),
                               np.delete(y, i))
        refit.append(y[i] - m.predict(x[i:i + 1])[0])
    mse = np.mean(loo_hat(x, y, a)[2] ** 2)
    mse2 = np.mean(np.square(refit))
    mse3 = cv.cv_results_[:, k].mean()
    print(f"{a:5.1f}   {mse:11.4f}   {mse2:8.4f}   {mse3:7.4f}")
print("alpha_ =", cv.alpha_, "  best_score_ =",
      round(cv.best_score_, 4))

try:
    RidgeCV(alphas=[0.0, 1.0]).fit(x, y)
except ValueError as err:
    print("ValueError:", err)
