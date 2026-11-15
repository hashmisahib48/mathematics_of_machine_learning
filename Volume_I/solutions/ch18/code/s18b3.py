# Problem 18B.3: priors as a lever (label 0 = malignant).
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA)
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
base = np.mean(y_tr == 0)             # share of malignant
print(f"{'prior':>7s} {'intercept_':>11s} {'missed':>7s} "
      f"{'false alarms':>13s} {'accuracy':>9s}  same w")
w0 = None
for pi in (base, 0.7, 0.9, 0.99):
    m = LDA(priors=[pi, 1 - pi]).fit(X_tr, y_tr)
    cm = confusion_matrix(y_te, m.predict(X_te))
    w0 = m.coef_ if w0 is None else w0
    same = np.allclose(m.coef_, w0)
    print(f"{pi:7.4f} {m.intercept_[0]:11.3f} {cm[0, 1]:7d} "
          f"{cm[1, 0]:13d} {m.score(X_te, y_te):9.4f}  {same}")
    # intercept_ minus ln(pi_benign / pi_malignant) is a constant
    print(f"{'':7s} intercept - ln((1-pi)/pi) = "
          f"{m.intercept_[0] - np.log((1 - pi) / pi):.3f}")
