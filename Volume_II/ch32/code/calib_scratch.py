# Platt scaling and temperature scaling from scratch.
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
def platt(f, y):                          # Eqs. 32.4-4 to 32.4-6
    n1 = y.sum(); n0 = len(y) - n1
    t = np.where(y == 1, (n1 + 1) / (n1 + 2), 1 / (n0 + 2))
    def loss(ab):                         # p = 1 / (1 + e^z)
        z = ab[0] * f + ab[1]
        p = 1 / (1 + np.exp(z))
        J = t * np.logaddexp(0, z) + (1 - t) * np.logaddexp(0, -z)
        return J.sum(), np.array([f @ (t - p), (t - p).sum()])
    ab0 = [0.0, np.log((n0 + 1) / (n1 + 1))]
    return minimize(loss, ab0, jac=True, method="L-BFGS-B").x

def temperature(f, y):                    # Eq. 32.4-7, two classes
    s = 2 * y - 1
    J = lambda lb: np.logaddexp(0, -2 * s * np.exp(lb) * f).sum()
    return np.exp(minimize_scalar(J, bounds=(-10, 10)).x)
X, y = load_breast_cancer(return_X_y=True)
svm = make_pipeline(StandardScaler(), SVC()).fit(X[:300], y[:300])
Xc, yc = X[300:], y[300:]              # held-out part
f = svm.decision_function(Xc)
out = {}
for m in ("sigmoid", "temperature"):      # the scikit-learn answers
    cal = CalibratedClassifierCV(FrozenEstimator(svm), method=m)
    cal.fit(Xc, yc)
    out[m[:3]] = (cal.calibrated_classifiers_[0].calibrators[0],
              cal.predict_proba(Xc)[:, 1])
A, B = platt(f, yc)
beta = temperature(f, yc)
g = out["sig"][0]
print("A, B :", np.round([A, B], 4), np.round([g.a_, g.b_], 4))
h = out["tem"][0]
print("beta :", round(beta, 4), round(h.beta_, 4))
d1 = np.abs(1 / (1 + np.exp(A * f + B)) - out["sig"][1]).max()
p2 = 1 / (1 + np.exp(-2 * beta * f))
d2 = np.abs(p2 - out["tem"][1]).max()
print(f"max |dp|: sigmoid {d1:.1e}, temperature {d2:.1e}")
assert max(d1, d2) < 1e-3
