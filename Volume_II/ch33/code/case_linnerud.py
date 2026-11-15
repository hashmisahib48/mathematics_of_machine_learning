# Case study: three body measurements from three exercises.
import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

d = load_linnerud()
X, Y = d.data, d.target
print("features:", d.feature_names, " targets:", d.target_names)
cv = KFold(5, shuffle=True, random_state=0)
ridge = Ridge(alpha=5.0)
models = {"mean of the training part": DummyRegressor(),
          "Ridge (native)": make_pipeline(StandardScaler(), ridge),
          "MultiOutputRegressor": make_pipeline(
              StandardScaler(), MultiOutputRegressor(ridge)),
          "RegressorChain": make_pipeline(
              StandardScaler(), RegressorChain(ridge)),
          "RandomForest (native)": RandomForestRegressor(
              min_samples_leaf=3, random_state=0)}
heads = "".join(f"{t:>8s}" for t in d.target_names)
print(f"{'5-fold R2':27s}{heads}{'all':>8s}")
for name, m in models.items():
    P = cross_val_predict(m, X, Y, cv=cv)
    r2 = r2_score(Y, P, multioutput="raw_values")
    print(f"{name:27s}" + "".join(f"{v:8.3f}" for v in r2)
          + f"{r2.mean():8.3f}")

Z = StandardScaler().fit_transform(X)
a = Ridge(alpha=5.0).fit(Z, Y)
m = MultiOutputRegressor(ridge).fit(Z, Y)
c = RegressorChain(ridge).fit(Z, Y)
w = np.vstack([e.coef_ for e in m.estimators_])
print("coef_ shape:", a.coef_.shape, " largest coef difference:",
      f"{abs(a.coef_ - w).max():.1e}")
print("chain against independent, largest prediction difference:",
      f"{abs(c.predict(Z) - m.predict(Z)).max():.1e}")
