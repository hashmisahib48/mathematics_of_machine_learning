# Problem 10B.3: changing the unit of one feature
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True)
X100 = X.copy()
X100[:, 2] *= 100                              # bmi in hundredths
tr, te = train_test_split(np.arange(len(y)), test_size=0.25,
                          random_state=0)  # split of Chapter 9

def run(name, model, data):
    model.fit(data[tr], y[tr])
    est = model[-1] if hasattr(model, "steps") else model
    r2 = model.score(data[te], y[te])
    w = est.coef_[2]
    print(f"{name:30s} w_bmi = {w:8.2f}  test R2 = {r2:.3f}")

run("least squares", LinearRegression(), X)
run("ridge 0.05", Ridge(alpha=0.05), X)
run("ridge 0.05, bmi x 100", Ridge(alpha=0.05), X100)
for lab, data in (("", X), (", bmi x 100", X100)):
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=0.05))
    run("scaled ridge 0.05" + lab, pipe, data)

for lab, data in (("original", X), ("bmi x 100", X100)):
    col = data[tr, 2] - data[tr, 2].mean()
    ss = col @ col
    print(f"{lab:10s} sum of squares: {ss:9.4f}"
          f"  ratio to alpha: {ss / 0.05:8.1f}")
