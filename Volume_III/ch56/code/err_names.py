# Section 56.7: the two commonest mistakes, caught.
from sklearn.model_selection import GridSearchCV
from common import cancer_split, svc_pipe

Xtr, Xte, ytr, yte = cancer_split()
try:
    GridSearchCV(svc_pipe(), {"C": [1, 10]}, cv=3).fit(Xtr, ytr)
except ValueError as e:
    print("ValueError:", str(e).split("Pipeline(")[0] + "Pipeline(...)")
try:
    from sklearn.model_selection import HalvingGridSearchCV
except ImportError as e:
    print("ImportError:", str(e).split(".")[0] + ".")
try:
    GridSearchCV(svc_pipe(), {"svc__C": [1, 10]}, cv=3,
                 scoring={"a": "accuracy", "f": "f1"}).fit(Xtr, ytr)
except ValueError as e:
    print("ValueError:", str(e)[:70] + "...")
