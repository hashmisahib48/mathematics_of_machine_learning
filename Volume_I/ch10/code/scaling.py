# Ridge is not invariant to units; a scaler in a pipeline fixes it.
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

km = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])
metres = 1000 * km

def report(name, model):
    p_km = model.fit(km, y).predict([[6.0]])[0]
    p_m = model.fit(metres, y).predict([[6000.0]])[0]
    print(f"{name:26s} 6 km: {p_km:7.3f}   6000 m: {p_m:7.3f}")

report("LinearRegression", LinearRegression())
report("Ridge(alpha=10)", Ridge(alpha=10.0))
report("scaler + Ridge(alpha=10)",
       make_pipeline(StandardScaler(), Ridge(alpha=10.0)))

w_km = Ridge(alpha=10.0).fit(km, y).coef_[0]
w_m = Ridge(alpha=10.0).fit(metres, y).coef_[0]
print(f"\nslope kept: {w_km / 1.7:.4f} of OLS (km), "
      f"{w_m * 1000 / 1.7:.6f} of OLS (m)")
