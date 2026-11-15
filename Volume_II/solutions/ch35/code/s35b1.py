# Problem 35B.1: components needed to reach an RMSE of 1.5.
import sys
sys.path.append("../../../ch35/code")  # chapter data
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from spectra import make_spectra

cv = KFold(5, shuffle=True, random_state=0)
def rmse(model, X, y):             # 5-fold CV RMSE
    s = cross_val_score(model, X, y, cv=cv,
                        scoring="neg_root_mean_squared_error")
    return -s.mean()

print("seed   k_PLS (RMSE)   k_PCR (RMSE)")
need = {"pls": [], "pcr": []}
for seed in range(100, 110):
    X, y = make_spectra(60, seed)
    out = []
    for name, make in (("pls", lambda k: PLSRegression(k)),
                       ("pcr", lambda k: make_pipeline(
                           StandardScaler(), PCA(k),
                           LinearRegression()))):
        for k in range(1, 13):
            r = rmse(make(k), X, y)
            if r <= 1.5:            # the smallest k that is good
                break
        need[name].append(k)
        out.append(f"{k:2d} ({r:.2f})")
    print(f"{seed:4d}   {out[0]:>10s}     {out[1]:>10s}")
for name in need:
    print(name, "median", np.median(need[name]), "all:", need[name])
