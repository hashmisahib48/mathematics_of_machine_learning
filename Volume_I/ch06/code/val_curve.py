# A validation curve: the penalty strength alpha as the knob.
import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

rng = np.random.default_rng(63)
X = rng.uniform(0, 1, (20, 1))                  # 20 samples
y = np.sin(2 * np.pi * X[:, 0])
y = y + rng.normal(0, 0.25, 20)                 # noise sigma = 0.25

model = make_pipeline(PolynomialFeatures(9, include_bias=False),
                      StandardScaler(), Ridge())
alphas = np.logspace(-8, 2, 11)                 # 1e-8 ... 1e2
tr, va = validation_curve(
    model, X, y, param_name="ridge__alpha", param_range=alphas,
    cv=5, scoring="neg_root_mean_squared_error")

print("alpha      train RMSE   validation RMSE")
for a, t, v in zip(alphas, -tr.mean(axis=1), -va.mean(axis=1)):
    print(f"{a:8.0e} {t:12.3f} {v:15.3f}")
best = alphas[np.argmax(va.mean(axis=1))]
print(f"best alpha on the validation folds: {best:.0e}")
