# scikit-learn's SGDRegressor on the diabetes data.
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor, LinearRegression

X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          random_state=0)
sgd = make_pipeline(StandardScaler(),
                    SGDRegressor(eta0=0.01, max_iter=1000, tol=1e-6,
                                 penalty=None, random_state=0))
ols = make_pipeline(StandardScaler(), LinearRegression())
for name, m in [("SGD", sgd), ("least squares", ols)]:
    m.fit(X_tr, y_tr)
    w = m[-1].coef_
    print(f"{name:13s} test R2 {m.score(X_te, y_te):.3f}"
          f"  w[:3] = {w[:3].round(2)}")
print("epochs used:", sgd[-1].n_iter_)
