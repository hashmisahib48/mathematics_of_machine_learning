# Read the solver names from the installed classes, then test them.
import textwrap
import warnings
import numpy as np
import scipy.sparse as sp
from sklearn import linear_model as lm
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

def names(cls, arg="solver"):          # allowed values
    out = []
    for c in cls._parameter_constraints[arg]:
        out += sorted(getattr(c, "options", []))
    return out

for cls in (lm.Ridge, lm.LogisticRegression, lm.PoissonRegressor,
            lm.QuantileRegressor, MLPClassifier):
    text = f"{cls.__name__}: " + ", ".join(names(cls))
    print(textwrap.fill(text, 66, subsequent_indent="    "))

X, y = make_classification(n_samples=200, n_features=5,
                           random_state=0)
for label, data, kw in [("dense", X, {}),
                        ("sparse", sp.csr_matrix(X), {}),
                        ("positive", X, {"positive": True})]:
    m = lm.Ridge(**kw).fit(data, y)
    print(f"Ridge auto, {label:9s} -> {m.solver_}")

print("\nLogisticRegression: penalty accepted by each solver")
cases = {"L2": {"l1_ratio": 0}, "L1": {"l1_ratio": 1},
         "elastic": {"l1_ratio": 0.5}, "none": {"C": np.inf}}
print(f"{'solver':16s}" + "".join(f"{k:>9s}" for k in cases))
for solver in names(lm.LogisticRegression):
    row = ""
    for kw in cases.values():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lm.LogisticRegression(solver=solver, max_iter=5000,
                                      **kw).fit(X, y)
            row += f"{'yes':>9s}"
        except ValueError:
            row += f"{'no':>9s}"
    print(f"{solver:16s}{row}")
