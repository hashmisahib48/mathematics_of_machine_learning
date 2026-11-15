# The deprecation of SVC(probability=True) in release 1.9.1.
import textwrap
import warnings
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    SVC(probability=True).fit(X, y)
print(w[0].category.__name__)
print(textwrap.fill(str(w[0].message), 66))
