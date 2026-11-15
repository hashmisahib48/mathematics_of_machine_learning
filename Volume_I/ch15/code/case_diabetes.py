# Case study: do curved features help on the diabetes data?
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import SplineTransformer, StandardScaler

data = load_diabetes()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data, data.target, test_size=0.25, random_state=0)
cv = KFold(5, shuffle=True, random_state=0)
print("training patients:", len(y_tr), " test patients:", len(y_te))
j = list(data.feature_names).index("bmi")

def poly(d):
    return PolynomialFeatures(d, include_bias=False)

spline = SplineTransformer(n_knots=5, include_bias=False)
alphas = np.logspace(-1, 4, 26)
lr = LinearRegression
models = [
    ("bmi: line", make_pipeline(lr()), [j]),
    ("bmi: degree 2", make_pipeline(poly(2), lr()), [j]),
    ("bmi: degree 3", make_pipeline(poly(3), lr()), [j]),
    ("bmi: spline", make_pipeline(spline, lr()), [j]),
    ("ten: line", make_pipeline(lr()), slice(None)),
    ("ten: degree 2 + ridge", make_pipeline(
        poly(2), StandardScaler(), RidgeCV(alphas=alphas)),
     slice(None))]
print(f"{'model':22s} columns  CV R2  test R2")
for name, m, c in models:
    A, B = X_tr[:, c], X_te[:, c]
    cv_r2 = cross_val_score(m, A, y_tr, cv=cv, scoring="r2")
    cv_r2 = cv_r2.mean()
    m.fit(A, y_tr)
    cols = A.shape[1]
    if len(m) > 1:
        cols = m[:-1].transform(A).shape[1]
    print(f"{name:22s} {cols:7d}  {cv_r2:.3f}  "
          f"{m.score(B, y_te):.3f}")
print("alpha chosen by RidgeCV:", round(float(m[-1].alpha_), 1))
