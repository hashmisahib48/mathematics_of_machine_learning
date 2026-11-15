# Problem 33B.1: normalised one-versus-rest probabilities on iris
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
Z = StandardScaler().fit_transform(X)
ovr = OneVsRestClassifier(LogisticRegression(max_iter=300)).fit(Z, y)
soft = LogisticRegression(max_iter=300).fit(Z, y)

np.set_printoptions(suppress=True)
i = 100                                    # the flower of the problem
z = Z[i:i + 1]
raw = np.array([e.predict_proba(z)[0, 1] for e in ovr.estimators_])
nrm = raw / raw.sum()
print("true class         :", y[i])
print("raw binary probs   :", raw.round(4),
      " sum", raw.sum().round(4))
print("normalised (33.3-3):", nrm.round(4))
print("OvR predict_proba  :", ovr.predict_proba(z)[0].round(4))
dif = nrm - ovr.predict_proba(z)[0]
print("max difference     :", np.abs(dif).max())
print("softmax proba      :", soft.predict_proba(z)[0].round(4))
print("training accuracy  : OvR %.4f, softmax %.4f"
      % (ovr.score(Z, y), soft.score(Z, y)))
