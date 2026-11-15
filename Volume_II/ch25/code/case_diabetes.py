# Case study: forests on the diabetes data (regression).
from sklearn.datasets import load_diabetes
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0)
kw = dict(n_estimators=300, random_state=0)
models = {
    "OLS": LinearRegression(),
    "one tree, leaf 20": DecisionTreeRegressor(
        min_samples_leaf=20, random_state=0),
    "forest, m = p": RandomForestRegressor(oob_score=True, **kw),
    "forest, m = p/3": RandomForestRegressor(
        max_features=0.33, oob_score=True, **kw),
    "forest, m = p/3, leaf 5": RandomForestRegressor(
        max_features=0.33, min_samples_leaf=5, oob_score=True,
        **kw),
    "extra trees, m = p/3, leaf 5": ExtraTreesRegressor(
        max_features=0.33, min_samples_leaf=5, **kw),
}
print("model                         5-fold  test R2  RMSE  OOB R2")
for name, m in models.items():
    cv = cross_val_score(m, X, y, cv=5).mean()   # as Chapter 9
    m.fit(X_tr, y_tr)
    rmse = mean_squared_error(y_te, m.predict(X_te)) ** 0.5
    oob = getattr(m, "oob_score_", None)
    oob = "     -" if oob is None else f"{oob:.4f}"
    print(f"{name:28s}  {cv:.4f}  {m.score(X_te, y_te):.4f}  "
          f"{rmse:5.1f} {oob}")
