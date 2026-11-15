# Section 57.8: regression scores in scikit-learn (diabetes).
import numpy as np
from sklearn import metrics as m
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      random_state=0)
fits = {"median": DummyRegressor(strategy="median"),
        "OLS": LinearRegression(), "HGB": HGB(random_state=0)}
fns = {"MAE": m.mean_absolute_error,
       "RMSE": m.root_mean_squared_error,
       "MedAE": m.median_absolute_error, "max": m.max_error,
       "MAPE": m.mean_absolute_percentage_error, "R2": m.r2_score,
       "EV": m.explained_variance_score,
       "D2abs": m.d2_absolute_error_score}
print(f"{'':7s}" + "".join(f"{k:>7s}" for k in fns))


def row(name, p):
    v = [f(yte, p) for f in fns.values()]
    print(f"{name:7s}" + "".join(
        f"{x:7.1f}" if abs(x) > 10 else f"{x:7.3f}" for x in v))


for name, est in fits.items():
    row(name, est.fit(Xtr, ytr).predict(Xte))
row("OLS+20", fits["OLS"].predict(Xte) + 20)   # a constant bias

rng = np.random.RandomState(0)                 # counts, Eq. 57.4-4
mu = np.exp(rng.normal(1.0, 0.8, 500))
c = rng.poisson(mu)
rates = {"true": mu, "shrunk": np.exp(1 + 0.5 * (np.log(mu) - 1)),
         "const": np.full(500, c.mean())}
for nm, pr in rates.items():
    d2 = m.d2_tweedie_score(c, pr, power=1)
    print(f"{nm:7s}D2 Poisson {d2:.4f}  R2 {m.r2_score(c, pr):.4f}")
