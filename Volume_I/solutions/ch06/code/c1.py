# Problem 6C.1: reading a learning curve; what ten times more data, or a cubic, would give.
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

def curve(degree, X, y, sizes):
    m = make_pipeline(PolynomialFeatures(degree, include_bias=False), LinearRegression())
    n, tr, va = learning_curve(m, X, y, train_sizes=sizes, cv=5, shuffle=True, random_state=0,
                               scoring="neg_root_mean_squared_error")
    return n, -tr.mean(axis=1), -va.mean(axis=1)

print("== the data of the figure: 300 samples, noise 0.25 (as in Listing 6.4)")
rng = np.random.default_rng(61); X = rng.uniform(0, 1, (300, 1)); y = np.sin(2 * np.pi * X[:, 0]) + rng.normal(0, 0.25, 300)
sizes = [15, 20, 30, 50, 80]
for d in (9, 3):
    n, tr, va = curve(d, X, y, sizes)
    print(f"degree {d}: n {n.tolist()}\n   train {tr.round(3).tolist()}\n   valid {va.round(3).tolist()}  gap {(va - tr).round(3).tolist()}")

print("\n== a larger pool: 3000 samples from the same source, train sizes up to 2400")
rng = np.random.default_rng(62); X = rng.uniform(0, 1, (3000, 1)); y = np.sin(2 * np.pi * X[:, 0]) + rng.normal(0, 0.25, 3000)
sizes = [15, 30, 80, 800, 2400]
for d in (9, 3):
    n, tr, va = curve(d, X, y, sizes)
    print(f"degree {d}: n {n.tolist()}  train {tr.round(3).tolist()}  valid {va.round(3).tolist()}")
xg = np.linspace(0, 1, 100001); c = np.polynomial.polynomial.polyfit(xg, np.sin(2 * np.pi * xg), 3)
b2 = np.mean((np.sin(2 * np.pi * xg) - np.polynomial.polynomial.polyval(xg, c)) ** 2)
print(f"limit n -> infinity: cubic sqrt(0.25^2 + bias^2) = {np.sqrt(0.0625 + b2):.4f} (bias^2 = {b2:.5f}); degree 9: 0.25 + tiny bias")
