# Case study 1: a seasonal series (72 months to learn, 24 ahead)
import json
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C, RBF, DotProduct, ExpSineSquared,
    WhiteKernel)

rng = np.random.default_rng(3)
t = np.arange(96) / 12.0                     # time in years
y = (2 + 0.5*t + 1.2*np.sin(2*np.pi*t) + 0.4*np.sin(4*np.pi*t)
     + 0.3*rng.normal(size=96))     # trend, season, noise
X, tr, te = t[:, None], slice(0, 72), slice(72, 96)

trend = C(1.0) * DotProduct(1.0)
season = C(1.0) * ExpSineSquared(1.0, 1.0) * RBF(5.0)
white = WhiteKernel(0.1)
models = {"trend + season + noise": trend + season + white,
          "RBF + noise": C(1.0) * RBF(1.0) + white}
out = {"t": t.tolist(), "y": y.tolist()}
for name, kern in models.items():
    gp = GaussianProcessRegressor(kern, normalize_y=True,
        n_restarts_optimizer=4, random_state=0)
    gp.fit(X[tr], y[tr])
    mean, std = gp.predict(X[te], return_std=True)
    rmse = np.sqrt(np.mean((mean - y[te])**2))
    cover = np.mean(np.abs(mean - y[te]) < 2*std)
    par = gp.kernel_.get_params()
    pick = lambda s: [v for k, v in par.items() if k.endswith(s)][0]
    noise_sd = np.sqrt(pick("noise_level")) * y[tr].std()
    print(f"{name}: LML {gp.log_marginal_likelihood_value_:.2f}")
    if "periodicity" in str(par):
        print(f"  period {pick('periodicity'):.3f} years")
    print(f"  noise sd {noise_sd:.3f}  test RMSE {rmse:.3f}  "
          f"2-sigma cover {cover:.2f}")
    m_all, s_all = gp.predict(X, return_std=True)
    out[name] = {"mean": m_all.tolist(), "std": s_all.tolist()}
base = np.sqrt(np.mean((y[te] - y[tr].mean())**2))
print(f"training-mean baseline: test RMSE {base:.3f}")
with open("../verify/case_seasonal.json", "w") as fh:
    json.dump(out, fh)
