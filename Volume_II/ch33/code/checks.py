# Numbers quoted in the text of Chapter 33 that no listing prints.
import numpy as np
from sklearn.base import clone
from sklearn.datasets import load_iris, load_digits, load_linnerud, make_blobs
from sklearn.linear_model import (LogisticRegression, RidgeClassifier, Ridge, LinearRegression)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier as OvO, OneVsRestClassifier as OvR, OutputCodeClassifier
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor, ClassifierChain, RegressorChain
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import hamming_loss, accuracy_score, f1_score
from sklearn.utils.multiclass import type_of_target

X, y = load_iris(return_X_y=True); Z = StandardScaler().fit_transform(X)
# 1. two classes: one binary problem
print("K=2: OvR estimators", len(OvR(LogisticRegression()).fit(Z[:100], y[:100]).estimators_),
      " OvO estimators", len(OvO(LogisticRegression()).fit(Z[:100], y[:100]).estimators_))
# 2. OvR predict_proba rows sum to one; multilabel rows do not
o = OvR(LogisticRegression(max_iter=300)).fit(Z, y)
print("OvR proba row sums min/max", o.predict_proba(Z).sum(1).min(), o.predict_proba(Z).sum(1).max())
Yl = np.c_[y == 0, y == 1, X[:, 0] > 5.8].astype(int)
ol = OvR(LogisticRegression(max_iter=300)).fit(Z, Yl)
print("multilabel_", ol.multilabel_, " proba row sums min/max", ol.predict_proba(Z).sum(1).min().round(3), ol.predict_proba(Z).sum(1).max().round(3))
print("OvR predict = argmax decision_function:", (o.predict(Z) == o.classes_[o.decision_function(Z).argmax(1)]).all())
# 3. LogisticRegression is softmax
lr = LogisticRegression(max_iter=300).fit(Z, y); d = lr.decision_function(Z); e = np.exp(d - d.max(1, keepdims=True))
print("softmax of decision_function equals predict_proba:", np.abs(e / e.sum(1, keepdims=True) - lr.predict_proba(Z)).max())
# 4. SVC and the OvO wrapper
for g in ["scale", 0.25]:
    s = SVC(gamma=g).fit(Z, y); w = OvO(SVC(gamma=g)).fit(Z, y)
    print("gamma", g, " max |SVC.decision_function(ovr) - OvO wrapper|:", round(float(np.abs(s.decision_function(Z) - w.decision_function(Z)).max()), 4), " same predictions:", (s.predict(Z) == w.predict(Z)).mean())
s = SVC(gamma=0.25, decision_function_shape="ovo").fit(Z, y); w = OvO(SVC(gamma=0.25)).fit(Z, y)
dw = np.array([e.decision_function(Z) for e in w.estimators_]).T
print("pair scores: wrapper = - SVC(ovo):", np.abs(dw + s.decision_function(Z)).max())
# 5. tie rule
cen = make_blobs(n_samples=90, centers=3, cluster_std=2.5, random_state=1)
Xb, yb = cen
a = SVC(kernel="linear").fit(Xb, yb); b = OvO(SVC(kernel="linear")).fit(Xb, yb); c = SVC(kernel="linear", break_ties=True).fit(Xb, yb)
G = np.random.default_rng(0).uniform(Xb.min(0) - 5, Xb.max(0) + 5, (200000, 2))
print("SVC vs wrapper differ on", int((a.predict(G) != b.predict(G)).sum()), "of 200000; wrapper vs break_ties differ on", int((b.predict(G) != c.predict(G)).sum()))
# 6. LinearSVC and OvR(LinearSVC)
Xd, yd = load_digits(return_X_y=True); Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.25, random_state=0, stratify=yd)
sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
A = LinearSVC().fit(Xtr, ytr); Bm = OvR(LinearSVC()).fit(Xtr, ytr)
print("LinearSVC vs OvR(LinearSVC): max coef diff", np.abs(A.coef_ - np.vstack([e.coef_ for e in Bm.estimators_])).max().round(4), " max |coef|", np.abs(A.coef_).max().round(3), " predictions equal:", (A.predict(Xte) == Bm.predict(Xte)).mean())
# 7. ECOC details
E = OutputCodeClassifier(LogisticRegression(max_iter=300), code_size=1.5, random_state=0).fit(Z, y)
print("ECOC K=3 code_size 1.5 -> bits", len(E.estimators_), " codes", sorted(set(E.code_book_.ravel())))
E2 = OutputCodeClassifier(KNeighborsClassifier(), code_size=2, random_state=0).fit(Z, y)
print("kNN base: code values", sorted(set(E2.code_book_.ravel())))
print("int(10*1.5), int(3*1.5):", int(10 * 1.5), int(3 * 1.5))
# 8. multioutput regression
L = load_linnerud(); XL, YL = L.data, L.target
a = Ridge(alpha=5.0).fit(XL, YL); m = MultiOutputRegressor(Ridge(alpha=5.0)).fit(XL, YL); r = RegressorChain(Ridge(alpha=5.0)).fit(XL, YL)
print("Ridge vs MultiOutputRegressor coef diff:", np.abs(a.coef_ - np.vstack([e.coef_ for e in m.estimators_])).max())
print("chain vs independent predictions diff:", np.abs(r.predict(XL) - m.predict(XL)).max())
# 9. type_of_target
print([type_of_target(t) for t in (["a", "b"], [0, 1, 2], Yl, YL, np.c_[y, y], [0.5, 1.2])])
# 10. what is inside the native estimators
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
g = GradientBoostingClassifier(n_estimators=5, random_state=0).fit(X, y)
print("GradientBoosting estimators_ shape:", g.estimators_.shape)
print("MLP out_activation_ multiclass:", MLPClassifier(max_iter=1000, random_state=0).fit(Z, y).out_activation_,
      " multilabel:", MLPClassifier(max_iter=1000, random_state=0).fit(Z, Yl).out_activation_)
print("0.9 ** 5 =", round(0.9 ** 5, 5))
# 11. masking belongs to the scorer: logistic scorers on the six points
Xs = np.array([[1.], [2], [4], [5], [7], [8]]); ys = np.array([0, 0, 1, 1, 2, 2])
for C in [1, 100, 1e4]:
    o = OvR(LogisticRegression(C=C, max_iter=2000)).fit(Xs, ys)
    print("OvR(LogisticRegression) C =", C, " predict:", o.predict(Xs))
o = OvR(LogisticRegression(C=100, max_iter=2000)).fit(Xs, ys)
print("C=100 decision values at x=4:", o.decision_function([[4.]]).round(2),
      " slopes", [round(float(e.coef_.item()), 2) for e in o.estimators_],
      " intercepts", [round(float(e.intercept_.item()), 2) for e in o.estimators_])
print("example 3 squared distances:", ((np.array([0.8, 0.9, -0.7, -0.6, 0.4, -0.9]) - np.array([[1, 1, 1, 1, 1, 1], [1, 1, -1, -1, -1, -1], [-1, -1, 1, 1, -1, -1], [-1, -1, -1, -1, 1, 1]])) ** 2).sum(1).round(2))
