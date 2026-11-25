# Selecting features before cross-validation: a leak.
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 5000))           # pure noise
y = rng.integers(0, 2, size=100)           # random classes
Xs = SelectKBest(k=20).fit_transform(X, y)  # uses all 100 labels
wrong = cross_val_score(LogisticRegression(), Xs, y, cv=5).mean()
pipe = make_pipeline(SelectKBest(k=20), LogisticRegression())
right = cross_val_score(pipe, X, y, cv=5).mean()
print(f"selection outside the folds: accuracy {wrong:.2f}")
print(f"selection inside a pipeline: accuracy {right:.2f}")
