# Choosing degree, knots and penalty by cross-validation.
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import SplineTransformer, StandardScaler

def curve(x):                          # a bump on a rising trend
    return 4 * np.exp(-(x - 3) ** 2 / 0.8) + 0.5 * x

rng = np.random.default_rng(0)
x = np.sort(rng.uniform(0, 10, 40))[:, None]
y = curve(x[:, 0]) + rng.normal(0, 0.4, 40)
x_new = rng.uniform(0, 10, 1000)[:, None]
y_new = curve(x_new[:, 0]) + rng.normal(0, 0.4, 1000)
cv = KFold(5, shuffle=True, random_state=0)

poly = make_pipeline(StandardScaler(), PolynomialFeatures(
    include_bias=False), LinearRegression())
spl = make_pipeline(SplineTransformer(include_bias=False),
                    LinearRegression())
spl20 = make_pipeline(SplineTransformer(n_knots=20,
                      include_bias=False), Ridge())
for model, name, values in (
        (poly, "polynomialfeatures__degree", range(1, 21)),
        (spl, "splinetransformer__n_knots", range(3, 21)),
        (spl20, "ridge__alpha", np.logspace(-5, 1, 13))):
    values = list(values)
    tr, va = validation_curve(
        model, x, y, param_name=name, param_range=values, cv=cv,
        scoring="neg_root_mean_squared_error")
    k = int(np.argmin(-va.mean(axis=1)))
    fit = model.set_params(**{name: values[k]}).fit(x, y)
    rmse = np.sqrt(np.mean((fit.predict(x_new) - y_new) ** 2))
    print(f"{name.split('__')[1]:8s} best = {values[k]:<8.4g}"
          f" CV RMSE = {-va.mean(axis=1)[k]:.3f}"
          f"  test RMSE = {rmse:.3f}")
