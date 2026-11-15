# Numbers quoted in the model answers to the four questions of Chapter 33.
import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.preprocessing import StandardScaler

print("== Q1  the tie-break term c/(div*(|c|+1))")
c = np.array([0.1, 1.0, 1.8, 10, 100, 1e4, 1e8])
for div in (3, 2, 1):
    t = c / (div * (np.abs(c) + 1)); print(f"divisor {div}: term at |c| = {c.tolist()}: {t.round(4).tolist()}  bound 1/{div}")
rng = np.random.default_rng(0); bad = {3: 0, 2: 0, 1: 0}
for _ in range(200000):               # worst case for one vote of difference: A has v+1 and c = -big, B has v and c = +big
    ca, cb = -abs(rng.standard_cauchy()) * 10, abs(rng.standard_cauchy()) * 10
    for div in bad:
        if 1 + ca / (div * (abs(ca) + 1)) < cb / (div * (abs(cb) + 1)): bad[div] += 1
print("random draws where a class with one vote more is overtaken:", bad)
ca, cb = -50.0, 50.0
for div in (1, 2, 3): print(f"divisor {div}: F_A - F_B for votes differing by one, c_A = -50, c_B = +50:", round(1 + ca / (div * 51) - cb / (div * 51), 4))

print("\n== Q2  1000 classes: one class with 0.1 per cent positives")
rng = np.random.default_rng(0); n = 20000
X = rng.normal(size=(n, 5)); logit = 1.5 * X[:, 0] + 1.0 * X[:, 1] - 8.0
y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
Xa, ya, Xb, yb = X[:10000], y[:10000], X[10000:], y[10000:]
print(f"share of events {y.mean():.4f}; in the training half {ya.sum()}, in the test half {yb.sum()}")
for cw in (None, "balanced", {0: 1, 1: 100}):
    m = LogisticRegression(class_weight=cw).fit(Xa, ya); p = m.predict(Xb); s = m.decision_function(Xb)
    print(f"class_weight={str(cw):22s} predicted positives {p.sum():4d}, true positives {int((p * yb).sum()):3d} of {yb.sum()}, intercept {m.intercept_[0]:+.2f}, mean score {s.mean():+.2f}")
print("accuracy of 'always no':", round(1 - yb.mean(), 4))

print("\n== Q3  chain trained on true labels, used with predicted labels")
def make(n, rng, noise):
    X = rng.uniform(-2, 2, (n, 2)); out = (X[:, 0] + noise * rng.normal(size=n)) > 0; beach = out & (X[:, 1] > 0)
    return X, np.c_[out, beach].astype(int)
base = LogisticRegression(C=1e4, max_iter=1000)
for noise in (0.0, 1.0, 2.0):
    r = {k: [] for k in ("independent", "chain cv=None", "chain cv=5")}; acc1 = []
    for seed in range(20):
        rg = np.random.default_rng(seed); Xtr, Ytr = make(300, rg, noise); Xte, Yte = make(2000, rg, noise)
        for k, m in (("independent", MultiOutputClassifier(base)), ("chain cv=None", ClassifierChain(base, order=[0, 1])), ("chain cv=5", ClassifierChain(base, order=[0, 1], cv=5))):
            P = m.fit(Xtr, Ytr).predict(Xte); r[k].append((P == Yte).all(1).mean())
            if k == "independent": acc1.append((P[:, 0] == Yte[:, 0]).mean())
    print(f"noise {noise}: accuracy of link 1 {np.mean(acc1):.4f};  subset accuracy " + ", ".join(f"{k} {np.mean(v):.4f}" for k, v in r.items()))

print("\n== Q4  wrapped and plain logistic regression on the digits")
X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
lr = LogisticRegression(max_iter=300); t = time.time(); soft = lr.fit(Xtr, ytr); ts = time.time() - t
t = time.time(); ovr = OneVsRestClassifier(LogisticRegression(max_iter=300)).fit(Xtr, ytr); to = time.time() - t
ps, po = soft.predict_proba(Xte), ovr.predict_proba(Xte)
print(f"test accuracy: softmax {soft.score(Xte, yte):.4f}, OvR {ovr.score(Xte, yte):.4f}")
print(f"test log loss: softmax {log_loss(yte, ps):.4f}, OvR (normalised) {log_loss(yte, po):.4f}")
print(f"test cases with a different predicted class: {(ps.argmax(1) != po.argmax(1)).sum()} of {len(yte)}")
print(f"mean largest probability: softmax {ps.max(1).mean():.4f}, OvR {po.max(1).mean():.4f}")
raw = np.column_stack([e.predict_proba(Xte)[:, 1] for e in ovr.estimators_])
print(f"raw OvR probabilities: row sums from {raw.sum(1).min():.3f} to {raw.sum(1).max():.3f}")
print(f"parameters: softmax {soft.coef_.size + soft.intercept_.size}, OvR {sum(e.coef_.size + e.intercept_.size for e in ovr.estimators_)}")
print(f"largest |coefficient|: softmax {np.abs(soft.coef_).max():.2f}, OvR {max(np.abs(e.coef_).max() for e in ovr.estimators_):.2f}")
