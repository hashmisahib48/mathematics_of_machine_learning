# One interface, many estimators: the same loop fits each model.
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          random_state=0)
models = [DummyRegressor(), LinearRegression(),
          KNeighborsRegressor(n_neighbors=10),
          DecisionTreeRegressor(max_depth=3, random_state=0)]
for m in models:
    m.fit(X_tr, y_tr)
    r2 = m.score(X_te, y_te)
    print(f"{type(m).__name__:22s} test R2 = {r2:6.3f}")
print("max_depth =", models[3].get_params()["max_depth"],
      " coef_ shape =", models[1].coef_.shape)
