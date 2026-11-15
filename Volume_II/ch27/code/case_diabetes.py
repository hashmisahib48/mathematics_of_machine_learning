# Diabetes: boosting against least squares (split of Chapter 9).
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)

data = load_diabetes()
Xtr, Xte, ytr, yte = train_test_split(
    data.data, data.target, test_size=0.25, random_state=0)


def report(name, model):
    cv = cross_val_score(model, Xtr, ytr, cv=5).mean()
    model.fit(Xtr, ytr)
    rmse = mean_squared_error(yte, model.predict(Xte)) ** 0.5
    print(f"{name:20s} cv {cv:.3f}  test R2 "
          f"{model.score(Xte, yte):.3f}  RMSE {rmse:.1f}")


report("least squares", LinearRegression())
report("default boosting", GBR(random_state=0))
stop = GBR(n_estimators=1000, n_iter_no_change=10, random_state=0)
report("early stopping", stop)
print("  trees kept by early stopping:", stop.n_estimators_)
grid = {"max_depth": [1, 2, 3], "learning_rate": [0.03, 0.1],
        "n_estimators": [100, 200], "subsample": [1.0, 0.5]}
gs = GridSearchCV(GBR(random_state=0), grid, cv=5).fit(Xtr, ytr)
p = gs.best_params_
print(f"grid search: depth {p['max_depth']}, rate "
      f"{p['learning_rate']}, {p['n_estimators']} trees, "
      f"subsample {p['subsample']}")
for loss in ("squared_error", "huber"):
    model = GBR(loss=loss, random_state=0, **p)
    report("tuned " + loss.split("_")[0], model)
imp = gs.best_estimator_.feature_importances_
top = np.argsort(imp)[::-1][:3]
print("largest importances:",
      {data.feature_names[k]: round(float(imp[k]), 3) for k in top})
