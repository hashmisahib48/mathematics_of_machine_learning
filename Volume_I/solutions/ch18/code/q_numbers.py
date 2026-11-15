# Numbers quoted in the model answers to the questions of Chapter 18
import warnings
import numpy as np
from scipy.stats import norm
from sklearn.datasets import load_breast_cancer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
np.set_printoptions(precision=4, suppress=True)

print("== Q1  where the boundary goes, and the error it minimises")
def err(t, m0, m1, s, p0, p1): return p0 * norm.sf((t - m0) / s) + p1 * norm.cdf((t - m1) / s)
m0, m1, v, p0, p1 = 2.0, 6.0, 2.0, 0.7, 0.3
xs = 4 + v / 4 * np.log(p0 / p1); s = np.sqrt(v)
print(f"means 2, 6, variance 2, priors 0.7/0.3: boundary {xs:.4f}; error at the midpoint 4: {err(4, m0, m1, s, p0, p1):.4f}; at x*: {err(xs, m0, m1, s, p0, p1):.4f}")
grid = np.linspace(2, 6, 4001); best = grid[np.argmin([err(t, m0, m1, s, p0, p1) for t in grid])]; print("numerical minimiser of the error:", round(best, 3))
print(f"always answering 'class 0': error {p1:.2f}")
for d in (4, 8, 16):
    print(f"means {d} apart, variance 2: shift of the boundary {v / d * np.log(p0 / p1):.4f}, in units of the gap {v / d ** 2 * np.log(p0 / p1):.4f}")
print(f"seeds (variance 1, means 3 and 7, priors 0.6/0.4): shift {1 / 4 * np.log(0.6 / 0.4):.4f}")

print("\n== Q2  LDA or QDA with few or many samples (p = 10, two classes)")
p = 10; idx = np.arange(p); C0 = 0.5 ** np.abs(idx[:, None] - idx[None, :]); L0 = np.linalg.cholesky(C0)
delta = 0.8 * np.ones(p)
def draw(nk, rng, unequal):
    X0 = rng.normal(size=(nk, p)) @ L0.T
    S1 = (2.0 * C0) if unequal else C0
    X1 = rng.normal(size=(nk, p)) @ np.linalg.cholesky(S1).T + delta
    return np.vstack([X0, X1]), np.r_[np.zeros(nk), np.ones(nk)].astype(int)
for unequal in (False, True):
    for nk in (25, 25000):
        acc = {"LDA": [], "QDA": []}
        for r in range(20 if nk == 25 else 3):
            rng = np.random.default_rng(100 + r); Xa, ya = draw(nk, rng, unequal); Xb, yb = draw(20000, rng, unequal)
            for nm, m in (("LDA", LDA()), ("QDA", QDA())): acc[nm].append(m.fit(Xa, ya).score(Xb, yb))
        print(f"class covariances {'different (x2)' if unequal else 'equal':16s} n_k = {nk:5d}:  LDA {np.mean(acc['LDA']):.3f}   QDA {np.mean(acc['QDA']):.3f}")
cl = lambda K, p: K * p + p * (p + 1) // 2 + K - 1; cq = lambda K, p: K * p + K * (p * (p + 1) // 2) + K - 1
print("parameters: LDA", cl(2, 10), " QDA", cq(2, 10), " for 50 samples")

print("\n== Q3  LDA and logistic regression on the breast-cancer training half")
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xte)
lda = LDA().fit(A, ytr); lr = LogisticRegression().fit(A, ytr)
wl, wg = lda.coef_[0], lr.coef_[0]
print("cosine of the two weight vectors:", round(float(wl @ wg / np.linalg.norm(wl) / np.linalg.norm(wg)), 3), " norms", round(float(np.linalg.norm(wl)), 2), round(float(np.linalg.norm(wg)), 2))
sl, sg = lda.decision_function(B), lr.decision_function(B)
print("test scores: LDA range", sl.min().round(1), sl.max().round(1), " logistic range", sg.min().round(1), sg.max().round(1), " correlation", round(float(np.corrcoef(sl, sg)[0, 1]), 3))
print("same predicted label on", int((lda.predict(B) == lr.predict(B)).sum()), "of", len(yte), " test accuracy LDA", round(lda.score(B, yte), 4), " logistic", round(lr.score(B, yte), 4))
print("log loss LDA", round(log_loss(yte, lda.predict_proba(B)), 4), " logistic", round(log_loss(yte, lr.predict_proba(B)), 4), "  tumours with |score| > 10:", int((np.abs(sl) > 10).sum()), "(LDA)", int((np.abs(sg) > 10).sum()), "(logistic)")
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for nm, est in (("LDA", LDA()), ("logistic", LogisticRegression())):
    s_ = cross_val_score(make_pipeline(StandardScaler(), est), Xtr, ytr, cv=cv); print(f"5-fold {nm}: {s_.mean():.4f} +- {s_.std():.4f}")

print("\n== Q4  a yes/no feature")
rng = np.random.default_rng(0)
def draw2(n):
    y = np.arange(n) % 2; x1 = rng.normal(size=n) + 1.0 * y; b = (rng.random(n) < np.where(y == 1, 0.9, 0.2)).astype(float)
    return np.c_[x1, b], y
Xa, ya = draw2(20000); Xb, yb = draw2(20000)
print("variance of the yes/no feature in the two classes: 0.2*0.8 =", 0.16, " 0.9*0.1 =", 0.09)
true_w = np.log(0.9 / 0.1) - np.log(0.2 / 0.8)
print(f"true log-odds weight of the yes/no feature ln[(0.9/0.1)/(0.2/0.8)] = {true_w:.3f}")
for nm, m in (("LDA", LDA()), ("logistic", LogisticRegression(C=1e6)), ("QDA", QDA())):
    m.fit(Xa, ya); w = m.coef_[0] if hasattr(m, "coef_") else None
    print(f"{nm:9s} weights {None if w is None else np.round(w, 3)}  accuracy {m.score(Xb, yb):.4f}  log loss {log_loss(yb, m.predict_proba(Xb)):.4f}")
