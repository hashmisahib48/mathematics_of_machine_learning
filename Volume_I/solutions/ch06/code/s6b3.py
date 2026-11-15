# Problem 6B.3: learning curves of a cubic at two noise levels.
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

sizes = [20, 40, 80, 160, 240]
p = 4                                          # weights of a cubic
xg = np.linspace(0, 1, 100001)                 # bias of best cubic
c = np.polynomial.polynomial.polyfit(xg, np.sin(2 * np.pi * xg), 3)
bias2 = np.mean((np.sin(2 * np.pi * xg)
                 - np.polynomial.polynomial.polyval(xg, c)) ** 2)
print(f"squared bias of the best cubic for sin(2 pi x): {bias2:.5f}")

for noise in (0.25, 0.5):
    rng = np.random.default_rng(61)
    X = rng.uniform(0, 1, (400, 1))
    y = np.sin(2 * np.pi * X[:, 0]) + rng.normal(0, noise, 400)
    model = make_pipeline(PolynomialFeatures(3, include_bias=False),
                          LinearRegression())
    n_used, tr, va = learning_curve(
        model, X, y, train_sizes=sizes, cv=5, shuffle=True,
        random_state=0, scoring="neg_root_mean_squared_error")
    print(f"\nnoise {noise}:  n   train   validation   theory")
    for n, a, b in zip(n_used, -tr.mean(axis=1), -va.mean(axis=1)):
        th = np.sqrt(noise**2 * (1 + p / n) + bias2)  # Eq. 6.4-3
        print(f"{n:17d} {a:7.3f} {b:10.3f} {th:9.3f}")
    se = noise / np.sqrt(800)      # error of an RMSE of 400 values
    print(f"sampling error of the RMSE: {se:.4f}")
