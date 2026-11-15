import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import (ARDRegression, BayesianRidge,
                                  LinearRegression, Ridge)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = load_diabetes()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data, data.target, test_size=0.25, random_state=0)

ols = LinearRegression().fit(X_tr, y_tr)
br = BayesianRidge().fit(X_tr, y_tr)
ard = ARDRegression().fit(X_tr, y_tr)
rd = Ridge(alpha=br.lambda_ / br.alpha_).fit(X_tr, y_tr)

print(f"alpha_ = {br.alpha_:.3e}  (noise sd {br.alpha_**-0.5:.1f})")
print(f"lambda_ = {br.lambda_:.3e}   n_iter_ = {br.n_iter_}")
ratio = br.lambda_ / br.alpha_
print(f"ridge strength lambda_/alpha_ = {ratio:.4f}")
print("OLS residual sd:",
      round(np.std(y_tr - ols.predict(X_tr), ddof=11), 1))
print("max |BayesianRidge - Ridge| weight:",
      round(np.abs(br.coef_ - rd.coef_).max(), 4))

for name, m in (("OLS", ols), ("BayesianRidge", br),
                ("ARDRegression", ard)):
    p = m.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, p))
    print(f"{name:14s} test R2 = {r2_score(y_te, p):.3f}"
          f"  RMSE = {rmse:.1f}")

mean, sd = br.predict(X_te, return_std=True)
z = np.abs(y_te - mean) / sd
print(f"predictive sd: min {sd.min():.1f}  max {sd.max():.1f}")
print(f"inside 1 sd: {np.mean(z < 1):.3f}   (nominal 0.683)")
print(f"inside 2 sd: {np.mean(z < 2):.3f}   (nominal 0.954)")
tiny = np.abs(ard.coef_) < 0.01
print("ARD weights below 0.01:",
      np.array(data.feature_names)[tiny].tolist())
print("ARD lambda_ of those:", np.round(ard.lambda_[tiny], 1))
