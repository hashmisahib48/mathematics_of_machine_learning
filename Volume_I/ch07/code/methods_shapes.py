# What each method returns, for a classifier and a transformer.
import numpy as np
from scipy.special import softmax
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.set_printoptions(suppress=True)
X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0)
sc = StandardScaler().fit(Xtr)                 # transformer: learn
Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)    # then apply
clf = LogisticRegression().fit(Ztr, ytr)       # predictor: learn
four = Zte[:4]                                 # four new samples
print("transform         :", four.shape)
print("predict           :", clf.predict(four).shape,
      clf.predict(four))
print("predict_proba     :", clf.predict_proba(four).shape)
print("decision_function :", clf.decision_function(four).shape)
print("score (accuracy)  :", round(clf.score(Zte, yte), 4))
same = StandardScaler().fit_transform(Xtr)     # fit + transform
print("fit_transform == fit then transform:",
      np.allclose(same, Ztr))
f = clf.decision_function(Zte[:1])[0]          # sample 0: scores
p = clf.predict_proba(Zte[:1])[0]
print("\nscores f       :", np.round(f, 3))
print("f - max(f)     :", np.round(f - f.max(), 3))
e = np.exp(f - f.max())
print("exp(f - max)   :", np.round(e, 5))
print("predict_proba  :", np.round(p, 5))
print("equals softmax :", np.allclose(p, softmax(f)))
print("rows sum to 1  :",
      np.allclose(clf.predict_proba(Zte).sum(axis=1), 1))