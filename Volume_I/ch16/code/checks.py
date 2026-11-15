# Facts quoted in the text of Chapter 16 (not a listing).
import warnings
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import LogisticRegressionCV as CV
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
Xb, yb = load_breast_cancer(return_X_y=True)
Xb = StandardScaler().fit_transform(Xb)


def msg(est, data=(Xb, yb)):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            est.fit(*data)
        except Exception as e:
            return f"{type(e).__name__}: {str(e)[:90]}"
    return "; ".join(f"{w.category.__name__}: {str(w.message)[:70]}"
                     for w in rec) or "no warning"


print("default LR           :", msg(LR()))
print("penalty=None         :", msg(LR(penalty=None)))
print("l1_ratio=None        :", msg(LR(l1_ratio=None)))
print("n_jobs=1             :", msg(LR(n_jobs=1)))
print("default CV           :", msg(CV()))
print("CV, 3 args set       :", msg(CV(l1_ratios=(0.0,), scoring="accuracy",
                                       use_legacy_attributes=False)))
print("CV penalty='l1'      :", msg(CV(penalty="l1", solver="liblinear",
                                       l1_ratios=(1.0,), scoring="accuracy",
                                       use_legacy_attributes=False)))
print("solver x penalty, binary (breast cancer) and 3 classes (iris)")
cases = {"L2": {}, "L1": dict(l1_ratio=1.0), "EN": dict(l1_ratio=0.5),
         "none": dict(C=np.inf)}
for data in ((Xb, yb), (X, y)):
    for solver in ("lbfgs", "liblinear", "newton-cg", "newton-cholesky",
                   "sag", "saga"):
        row = []
        for name, kw in cases.items():
            r = msg(LR(solver=solver, max_iter=3000, **kw), data)
            row.append("yes" if "Error" not in r else "no")
        print(f"  {len(set(data[1]))} classes {solver:16s}", row)
m3 = LR(max_iter=1000).fit(X, y)
print("multi-class coef_", m3.coef_.shape, " intercept_", m3.intercept_.shape)

print("--- labels, classes_ and columns of predict_proba")
print("breast cancer counts", np.bincount(yb),
      load_breast_cancer().target_names)
xs = np.array([[1.0], [2.0], [3.0], [4.0]])
lab = np.array(["fail", "fail", "pass", "pass"])
mm = LR().fit(xs, lab)
print("string labels: classes_", mm.classes_, " P(row of x=4) =",
      np.round(mm.predict_proba([[4.0]]), 3))
print("labels 1 and 2: classes_", LR().fit(xs, [1, 1, 2, 2]).classes_)
cvm = CV(l1_ratios=(0.0,), scoring="accuracy",
         use_legacy_attributes=False).fit(Xb, yb)
print("default Cs_:", cvm.Cs_.round(5), cvm.Cs_.shape)
print("--- an extra student who studied 20 hours and passed")
x8 = np.arange(1.0, 9.0)
y8 = np.array([0, 0, 0, 1, 0, 1, 1, 1])
for extra in (False, True):
    xx, yy = (np.r_[x8, 20.0], np.r_[y8, 1]) if extra else (x8, y8)
    ols = np.polyfit(xx, yy, 1)
    lg = LR(C=np.inf, tol=1e-10, max_iter=1000).fit(xx[:, None], yy)
    print(f"extra={extra}: OLS slope {ols[0]:.4f}, 0.5 at x ="
          f" {(0.5 - ols[1]) / ols[0]:.3f};  logistic w"
          f" {lg.coef_[0, 0]:.4f}, boundary"
          f" {-lg.intercept_[0] / lg.coef_[0, 0]:.3f}")
