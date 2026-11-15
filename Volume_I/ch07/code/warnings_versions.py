# Versions, and the warning a beginner meets first.
import warnings
import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

print("scikit-learn", sklearn.__version__)
print("numpy       ", np.__version__)
X, y = load_iris(return_X_y=True, as_frame=True)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    LogisticRegression().fit(X, y)            # unscaled features
print("raw features :", caught[0].category.__name__)
print(" ", caught[0].message.args[0].splitlines()[0])
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(X, y)                           # scaled
print("scaled       :", len(caught), "warnings")