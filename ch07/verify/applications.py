# Numbers quoted in the text of Sections 7.4 and 7.6 to 7.7 (run: python3 applications.py).
import os, sys, math, warnings
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
from logistic import LogisticRegression as Mine
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
X = [[1], [2], [3], [4], [5], [6]]; y = [0, 0, 1, 0, 1, 1]
m = Mine(eta=0.5, n_iter=20000, tol=1e-8).fit(X, y); w, b = m.coef_[0], m.intercept_
t = b + 5 * w
print(f"7.4-2 w {w:.4f} b {b:.4f} e^w {math.exp(w):.3f} boundary {-b / w:.4f}")
print(f"7.4-2 t(5) {t:.4f} P {1 / (1 + math.exp(-t)):.4f} sum p {sum(m.predict_proba(X)):.4f} sum y {sum(y)}")
tr = -4.2491 + 1.2140 * 5; print(f"7.4-2 from the rounded values t(5) {tr:.4f} P {1 / (1 + math.exp(-tr)):.4f}")
print("7.4-2 misclassified students", [i + 1 for i, (p, c) in enumerate(zip(m.predict(X), y)) if p != c])
D = load_breast_cancer(); a, bb, c, d = train_test_split(D.data, D.target, test_size=0.25, random_state=0, stratify=D.target)
lr = make_pipeline(StandardScaler(), LogisticRegression()).fit(a, c)
kn = make_pipeline(StandardScaler(), KNeighborsClassifier(5)).fit(a, c)
print("7.7 test tumors", len(d), "malignant", int((d == 0).sum()), "benign", int((d == 1).sum()), "training", len(c))
print("7.7 correct: logistic", int((lr.predict(bb) == d).sum()), "5-NN", int((kn.predict(bb) == d).sum()),
      "numbers stored by logistic", lr[-1].coef_.size + 1, "by k-NN", len(c) * a.shape[1])
w4 = lr[-1].coef_[0]; j = int(np.argsort(-np.abs(w4))[0])
print("7.7 largest weight", D.feature_names[j], round(w4[j], 4), "odds x", round(math.exp(w4[j]), 4), "inverse", round(math.exp(-w4[j]), 3))
pm = lr.predict_proba(bb)[:, 0]
for th in (0.5, 0.1, 0.02):
    al = pm >= th; print("7.7 th", th, "missed", int(((d == 0) & ~al).sum()), "false", int(((d == 1) & al).sum()))
X4, y4 = load_iris(return_X_y=True)
ir = make_pipeline(StandardScaler(), LogisticRegression(C=10)).fit(X4, y4); W = ir[-1].coef_
print("7.7 iris petal width weights", W[:, 3].round(3), "virginica vs versicolor odds x", round(math.exp(W[2, 3] - W[1, 3]), 1))
print("7.7 iris petal length weights", W[:, 2].round(3), "setosa vs versicolor", round(math.exp(W[0, 2] - W[1, 2]), 3))
print("7.7 flower 70 scores", ir.decision_function(X4[[70]]).round(3), "P", ir.predict_proba(X4[[70]]).round(3), "ratio", round(ir.predict_proba(X4[[70]])[0, 2] / ir.predict_proba(X4[[70]])[0, 1], 2))
Xa = np.array([[1], [2], [3], [4], [5], [6]]); ya = np.array(y)
for kw in ({"l1_ratio": 0.5}, {"l1_ratio": 0.5, "solver": "saga"}):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); LogisticRegression(**kw).fit(Xa, ya)
        print("7.6 options", kw, "runs")
    except Exception as e: print("7.6 options", kw, "error:", str(e)[:60])
with warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter("always"); LogisticRegression(penalty="l2").fit(Xa, ya)
print("7.6 penalty argument warns:", any("deprecated" in str(x.message) for x in ws))
