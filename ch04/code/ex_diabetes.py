# Many features: quadratic diabetes model, with and without ridge.
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge

X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          random_state=0)
quad = PolynomialFeatures(degree=2, include_bias=False)
print("features after expansion:", quad.fit_transform(X).shape[1])
for name, est in [("linear, 10 features", None),
                  ("quadratic, LS", LinearRegression()),
                  ("quadratic, ridge 10", Ridge(alpha=10.0)),
                  ("quadratic, ridge 100", Ridge(alpha=100.0))]:
    if est is None:
        m = LinearRegression()
    else:
        m = make_pipeline(PolynomialFeatures(2, include_bias=False),
                          StandardScaler(), est)
    m.fit(X_tr, y_tr)
    a, b = m.score(X_tr, y_tr), m.score(X_te, y_te)
    print(f"{name:21s} train R2 {a:.3f}  test R2 {b:.3f}")
