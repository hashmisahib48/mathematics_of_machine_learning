# Numbers quoted in the model answers to Questions 1 - 4.  Run: python3 code/q_numbers.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../ch26/code'))
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier as DTC

print("== Q1  the six apples")
x = np.arange(1, 7); y = np.array([-1, -1, -1, 1, -1, 1])
h1 = np.where(x <= 3.5, -1, 1)
w2 = np.array([1, 1, 1, 1, 5, 1]) / 10          # weights of round 2 (Example 26.5-1)
print("h1 wrong on apples", (x[h1 != y]).tolist(), " eps1 = 1/6 ->", round(np.log(5), 4))
print("weighted error of h1 under the round-2 weights:", w2[h1 != y].sum().round(4), "-> alpha =", round(np.log((1 - .5) / .5), 4))
print("reversed h1 has error", round(1 - 1 / 6, 4), "alpha =", round(np.log((1 / 6) / (5 / 6)), 4), "= minus", round(np.log(5), 4))
for e in (0.2, 0.8): print(f"eps={e}: alpha={np.log((1-e)/e):+.4f}  e^alpha={(1-e)/e:.4f}")

print("\n== Q2  bias and variance, breast cancer, training part")
cd = load_breast_cancer()
Xtr, Xte, ytr, yte = train_test_split(cd.data, cd.target, test_size=0.25, random_state=0, stratify=cd.target)
models = {
    "one stump": lambda: DTC(max_depth=1, random_state=0),
    "one full tree": lambda: DTC(random_state=0),
    "forest, 100 full trees": lambda: RandomForestClassifier(100, random_state=0),
    "forest, 100 stumps": lambda: RandomForestClassifier(100, max_depth=1, random_state=0),
    "AdaBoost, 50 stumps": lambda: AdaBoostClassifier(random_state=0),
}
rng = np.random.default_rng(0); halves = [rng.permutation(len(ytr)) for _ in range(10)]
print(f"{'model':24s} train   test   (halves)  disagreement")
for name, mk in models.items():
    full = mk().fit(Xtr, ytr); accs, dis = [], []
    for p in halves:
        a, b = p[:len(p) // 2], p[len(p) // 2:]
        ma, mb = mk().fit(Xtr[a], ytr[a]), mk().fit(Xtr[b], ytr[b])
        accs += [ma.score(Xte, yte), mb.score(Xte, yte)]
        dis.append((ma.predict(Xte) != mb.predict(Xte)).mean())
    print(f"{name:24s} {full.score(Xtr, ytr):.4f} {full.score(Xte, yte):.4f}   {np.mean(accs):.4f}    {np.mean(dis):.4f}  ({100*np.mean(dis):.2f} per cent)")
ada = AdaBoostClassifier(DTC(max_depth=None), random_state=0).fit(Xtr, ytr)
print("AdaBoost on full trees keeps", len(ada.estimators_), "round(s); test", round(ada.score(Xte, yte), 4))

print("\n== Q3  the weight of flipped labels, stumps, 10 repetitions")
R, N = 10, 300; rounds = (1, 2, 5, 10, 25, 50, 100, 200, 400); S, Smax, C, SM = [], [], [], []
for rep in range(R):
    X, yy = make_moons(N + 1000, noise=0.3, random_state=rep); Xa, ya, Xb, yb = X[:N], yy[:N], X[N:], yy[N:]
    flip = np.random.default_rng(rep).random(N) < 0.10; yn = np.where(flip, 1 - ya, ya)
    m = AdaBoostClassifier(DTC(max_depth=1), n_estimators=400, random_state=0).fit(Xa, yn)
    w = np.full(N, 1 / N); s, ms = [], []
    for e, a in zip(m.estimators_, m.estimator_weights_):
        wr = e.predict(Xa) != yn; w = w * np.exp(a * wr); w /= w.sum()      # weights after the update
        s.append(w[flip].sum()); ms.append(w[flip & wr].sum())  # share, and the part of it that is a mistake
    S.append(s); SM.append(ms)
    C.append((1 - np.array(list(m.staged_score(Xb, yb)))))
S, SM, C = np.mean(S, 0), np.mean(SM, 0), np.mean(C, 0)
print("rounds                      ", *[f"{r:6d}" for r in rounds])
print("weight of flipped samples   ", *[f"{S[r-1]:6.3f}" for r in rounds])
print("... of which mistakes of h_m  ", *[f"{SM[r-1]:6.3f}" for r in rounds])
print("test error on clean data    ", *[f"{C[r-1]:6.3f}" for r in rounds])
print("largest share over rounds:", S.max().round(3), "at round", int(S.argmax()) + 1, "; flipped samples are 0.10 of the data")

print("\n== Q4  accuracy against log loss, breast cancer")
lr = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xtr, ytr)
cands = (("logistic regression", lr),
         ("AdaBoost, 200 stumps", AdaBoostClassifier(n_estimators=200, random_state=0).fit(Xtr, ytr)),
         ("AdaBoost, 50 trees of depth 3", AdaBoostClassifier(DTC(max_depth=3), random_state=0).fit(Xtr, ytr)),
         ("  same, sigmoid-calibrated", CalibratedClassifierCV(AdaBoostClassifier(DTC(max_depth=3), random_state=0), method="sigmoid", cv=5).fit(Xtr, ytr)),
         ("  200 stumps, sigmoid-calibrated", CalibratedClassifierCV(AdaBoostClassifier(n_estimators=200, random_state=0), method="sigmoid", cv=5).fit(Xtr, ytr)))
for name, m in cands:
    p = m.predict_proba(Xte)
    print(f"{name:31s} accuracy {m.score(Xte, yte):.4f}  AUC {roc_auc_score(yte, p[:, 1]):.4f}  log loss {log_loss(yte, p):.4f}  largest p {p.max():.4f}")
print("log loss of a classifier that is always right but sits at the largest possible AdaBoost probability 0.8808:", round(-np.log(0.8808), 4))
