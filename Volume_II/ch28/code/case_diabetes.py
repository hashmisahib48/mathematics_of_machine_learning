# Case study: HistGradientBoostingRegressor on the diabetes data.
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split

X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0)

def rmse(m):
    return np.sqrt(np.mean((m.predict(X_te) - y_te) ** 2))

grid = {"max_leaf_nodes": [3, 8], "min_samples_leaf": [10, 20],
        "learning_rate": [0.03, 0.1], "max_iter": [50, 100, 200]}
models = {"OLS": LinearRegression().fit(X_tr, y_tr),
          "HGB default": HGB(random_state=0).fit(X_tr, y_tr),
          "HGB tuned": GridSearchCV(HGB(random_state=0), grid,
                                    cv=5).fit(X_tr, y_tr)}
for name, m in models.items():
    print("%-12s test R2 %.3f  RMSE %.1f" % (
        name, m.score(X_te, y_te), rmse(m)))
print("tuned:", models["HGB tuned"].best_params_)
