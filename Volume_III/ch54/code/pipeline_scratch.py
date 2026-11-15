# A minimal Pipeline, checked against scikit-learn's.
import numpy as np
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class MiniPipeline:
    def __init__(self, steps):
        self.steps = steps                    # a list of estimators

    def fit(self, X, y):
        self.fitted_ = []
        for est in self.steps[:-1]:
            est = clone(est)                  # theta_j, Eq. 54.3-1
            X = est.fit_transform(X, y)       # X_j from X_(j-1)
            self.fitted_.append(est)
        self.fitted_.append(clone(self.steps[-1]).fit(X, y))
        return self
    def predict_proba(self, X):
        for est in self.fitted_[:-1]:
            X = est.transform(X)              # stored numbers only
        return self.fitted_[-1].predict_proba(X)
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

X, y = load_breast_cancer(return_X_y=True)
parts = [StandardScaler(), PCA(5, random_state=0),
         LogisticRegression(max_iter=1000)]
mine = MiniPipeline(parts).fit(X[:400], y[:400])
ref = make_pipeline(*parts).fit(X[:400], y[:400])
p1, p2 = mine.predict_proba(X[400:]), ref.predict_proba(X[400:])
assert (mine.predict(X[400:]) == ref.predict(X[400:])).all()
assert np.allclose(p1, p2, atol=1e-12)
print("labels equal; largest probability difference",
      f"{np.abs(p1 - p2).max():.1e}")
