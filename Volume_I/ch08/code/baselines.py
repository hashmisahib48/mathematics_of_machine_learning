# 8.2  Baselines: diabetes (regression), digit 9 or not (classes).
from sklearn.datasets import load_diabetes, load_digits
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0)
for name, m in (("mean", DummyRegressor(strategy="mean")),
                ("median", DummyRegressor(strategy="median")),
                ("quantile 0.9", DummyRegressor(
                    strategy="quantile", quantile=0.9))):
    p = m.fit(X_tr, y_tr).predict(X_te)
    print(f"{name:12s} constant_ = {m.constant_[0, 0]:7.2f}"
          f"  test R2 = {r2_score(y_te, p):7.3f}"
          f"  RMSE = {root_mean_squared_error(y_te, p):5.1f}")

X, y = load_digits(return_X_y=True)
y = (y == 9).astype(int)               # 1 = nine, 0 = another digit
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
print("fraction of nines:", round(y.mean(), 4))
for s in ("most_frequent", "prior", "stratified", "uniform"):
    m = DummyClassifier(strategy=s, random_state=0).fit(X_tr, y_tr)
    print(f"{s:14s} accuracy = {m.score(X_te, y_te):.3f}"
          f"   share predicted nine = {m.predict(X_te).mean():.3f}")
for s in ("prior", "most_frequent"):
    m = DummyClassifier(strategy=s).fit(X_tr, y_tr)
    print(f"{s:13s} predict_proba:", m.predict_proba(X_te[:1])[0])
