import numpy as np
from sklearn.linear_model import Ridge
from clinic import X

try:                                  # not yet enabled
    from sklearn.impute import IterativeImputer
except ImportError as e:
    print("ImportError:", str(e)[:68], "...")

y = np.arange(6.0)
try:
    Ridge().fit(X, y)
except ValueError as e:
    print("ValueError:", " ".join(str(e).split()[:11]), "...")

from sklearn.impute import MissingIndicator
mi = MissingIndicator().fit(X[:, [0, 1]][[0, 5, 2, 4]])   # no NaN in age
try:
    mi.transform(X[:, [0, 1]])
except ValueError as e:
    print("ValueError:", str(e).splitlines()[0])
