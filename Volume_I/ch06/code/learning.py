# Learning curves for three model complexities.
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(61)
X = rng.uniform(0, 1, (300, 1))            # 300 samples
y = np.sin(2 * np.pi * X[:, 0])
y = y + rng.normal(0, 0.25, 300)           # noise sigma = 0.25
sizes = np.array([15, 20, 30, 50, 80, 120, 200])  # train sizes

rows = {}
for degree in (1, 3, 9):
    model = make_pipeline(
        PolynomialFeatures(degree, include_bias=False),
        LinearRegression())
    n_used, tr, va = learning_curve(
        model, X, y, train_sizes=sizes, cv=5, shuffle=True,
        random_state=0,
        scoring="neg_root_mean_squared_error")
    rows[degree] = (-tr.mean(axis=1), -va.mean(axis=1))
print("scores have shape", tr.shape, "= (sizes, folds)")
print("        degree 1       degree 3       degree 9")
print("  n   train   val    train   val    train   val")
for k, n in enumerate(n_used):
    cells = "".join(f"  {rows[d][0][k]:.3f} {rows[d][1][k]:.3f} "
                    for d in (1, 3, 9))
    print(f"{n:4d} {cells}")
