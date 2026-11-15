# Numbers quoted in the model answers to the questions of Chapter 22.
import sys, time, numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression

X8 = np.array([[1, 5], [2, 3], [3, 4], [4, 1], [7, 6], [8, 2], [4, 7], [6, 3]], float); y8 = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
hard = lambda X, y: SVC(kernel="linear", C=1e6, tol=1e-10).fit(X, y)
m0 = hard(X8, y8); w0, b0 = m0.coef_[0], float(m0.intercept_[0])
print("== Q1  eight seeds: w, b, width", w0.round(4), round(b0, 4), round(2 / np.linalg.norm(w0), 4))
keep = np.arange(8) != 4; m = hard(X8[keep], y8[keep])
print("seed 5 (alpha = 0) removed : w, b", m.coef_[0].round(4), round(float(m.intercept_[0]), 4))
X = X8.copy(); X[4] = [20, 20]; m = hard(X, y8)
print("seed 5 moved to (20, 20)   : w, b", m.coef_[0].round(4), round(float(m.intercept_[0]), 4))
X = X8.copy(); X[4] = [6, 5]; m = hard(X, y8)
print("seed 5 moved to (6, 5), y f of the old line = %.2f : w, b" % (w0 @ [6, 5] + b0), m.coef_[0].round(4), round(float(m.intercept_[0]), 4))
for s_, new in ((8, [6, 4]), (7, [4, 8])):
    X = X8.copy(); X[s_ - 1] = new; m = hard(X, y8)
    print("support vector %d moved to %s: w, b, width" % (s_, new), m.coef_[0].round(4), round(float(m.intercept_[0]), 4), round(2 / np.linalg.norm(m.coef_), 4))
m = hard(X8[np.arange(8) != 7], y8[np.arange(8) != 7])
print("support vector 8 removed: width", round(2 / np.linalg.norm(m.coef_), 4))

print("== Q2  C times 100")
bc = load_breast_cancer(); Xtr, Xte, ytr, yte = train_test_split(bc.data, bc.target, test_size=0.25, random_state=0, stratify=bc.target)
Z = StandardScaler().fit(Xtr).transform(Xtr)
for C in (0.01, 1.0, 100.0):
    mm = SVC(kernel="linear", C=C).fit(Z, ytr); print("breast cancer, linear, C = %-6g SV %3d  width %.3f" % (C, len(mm.support_), 2 / np.linalg.norm(mm.coef_)))
for C in (0.1, 10.0):
    mm = SVC(kernel="linear", C=C, tol=1e-10).fit(X8, y8); print("eight seeds,    linear, C = %-6g SV %3d  width %.3f" % (C, len(mm.support_), 2 / np.linalg.norm(mm.coef_)))
a = SVC(kernel="linear", C=0.2, tol=1e-12).fit(X8, y8)
b = SVC(kernel="linear", C=0.02, tol=1e-12).fit(np.tile(X8, (10, 1)), np.tile(y8, 10))
print("ten copies of every seed with C/10 = 0.02 against one copy with C = 0.2: max |dw| = %.1e, |db| = %.1e" % (np.abs(a.coef_ - b.coef_).max(), abs(a.intercept_[0] - b.intercept_[0])))
c = SVC(kernel="linear", C=0.2, tol=1e-12).fit(np.tile(X8, (10, 1)), np.tile(y8, 10))
print("ten copies with the old C = 0.2: width %.3f against %.3f" % (2 / np.linalg.norm(c.coef_), 2 / np.linalg.norm(a.coef_)))

print("== Q3  gamma at fixed C = 1 (the data of Figure 22.11-1)")
Xm, ym = make_moons(n_samples=300, noise=0.28, random_state=0)
Xa, Xb, ya, yb = train_test_split(Xm, ym, test_size=0.5, random_state=0, stratify=ym)
for g in (0.1, 1, 10, 100, 1000):
    mm = SVC(C=1, gamma=g).fit(Xa, ya); print("gamma %-6g SV %3d  train %.3f  test %.3f" % (g, len(mm.support_), mm.score(Xa, ya), mm.score(Xb, yb)))

print("== Q4  the three numbers")
Xtr_s = StandardScaler().fit(Xtr).transform(Xtr)
def stats(Xa_, gam):
    d2 = ((Xa_[:, None, :] - Xa_[None, :, :])**2).sum(-1); iu = np.triu_indices(len(Xa_), 1)
    return gam * d2[iu].mean(), np.exp(-gam * d2)[iu].mean()
p = Xtr.shape[1]
for nm, Xa_, gam in (("raw scale", Xtr, 1 / (p * Xtr.var())), ("raw auto", Xtr, 1 / p), ("standardised", Xtr_s, 1 / (p * Xtr_s.var()))):
    s1, s2 = stats(Xa_, gam); print("%-13s gamma %.3e  mean gamma d2 %10.2f  mean k(x,z) over pairs %.4f" % (nm, gam, s1, s2))
m_auto = SVC(gamma="auto").fit(Xtr, ytr); pr = m_auto.predict(Xte)
print("raw, gamma auto: predicted classes", np.unique(pr), " share of benign in the test set %.4f  test accuracy %.4f  support vectors %d of %d" % ((yte == 1).mean(), m_auto.score(Xte, yte), len(m_auto.support_), len(ytr)))

print("== Q5  calibration")
t = time.perf_counter(); plain = make_pipeline(StandardScaler(), SVC()).fit(Xtr, ytr); t1 = time.perf_counter() - t
t = time.perf_counter(); cal = make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(), method="sigmoid", cv=5, ensemble=False)).fit(Xtr, ytr); t2 = time.perf_counter() - t
f = plain.decision_function(Xte); pc = cal.predict_proba(Xte)[:, 1]
print("decision values on the test set: min %.2f max %.2f" % (f.min(), f.max()))
sig = cal[-1].calibrated_classifiers_[0].calibrators[0]
print("Platt A, B = %.2f, %.2f" % (sig.a_, sig.b_), " P(f=0) = %.3f  P(f=1) = %.3f  P(f=2) = %.3f" % tuple(1 / (1 + np.exp(sig.a_ * v + sig.b_)) for v in (0, 1, 2)))
print("fit time: plain %.3f s, calibrated %.3f s, ratio %.1f" % (t1, t2, t2 / t1))
print("accuracy plain %.4f calibrated %.4f; log loss calibrated %.4f" % (plain.score(Xte, yte), cal.score(Xte, yte), log_loss(yte, pc)))
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr); print("logistic regression log loss %.4f" % log_loss(yte, lr.predict_proba(Xte)[:, 1]))
