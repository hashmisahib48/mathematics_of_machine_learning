# Numbers quoted in the model answers to the questions of Chapter 57
import sys, os
import numpy as np
from scipy.stats import norm
from sklearn import metrics as M
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE + '/../../../ch57/code')
from screendata import make_screen

print("== Q1  fraud detector, 99 % accuracy, 0.5 % fraud, 10000 transactions")
n = 10000; fraud = 50
print("rule 'no fraud': accuracy", (n - fraud) / n, " errors", fraud)
for name, TP, FN, FP in (("A", 0, 50, 50), ("B", 40, 10, 90), ("C", 25, 25, 75)):
    TN = n - TP - FN - FP
    y = np.r_[np.ones(TP + FN), np.zeros(FP + TN)]
    p = np.r_[np.ones(TP), np.zeros(FN), np.ones(FP), np.zeros(TN)]
    prec = TP / (TP + FP) if TP + FP else float("nan")
    print(f"detector {name}: TP {TP} FN {FN} FP {FP} TN {TN}  accuracy {(TP + TN) / n:.4f}"
          f"  recall {TP / 50:.3f}  precision {prec:.4f}  MCC {M.matthews_corrcoef(y, p):.4f}"
          f"  F1 {M.f1_score(y, p):.4f}")
print("a false alarm costs 1, a missed fraud costs 100: cost of 'no fraud'", fraud * 100,
      " detector B", 10 * 100 + 90 * 1)

print("\n== Q2  prevalence, AP and AUC (screen of Section 57.1, d = 2, threshold 1.5)")
d, t = 2.0, 1.5
tpr, fpr = norm.sf(t - d), norm.sf(t)
print("TPR %.4f FPR %.4f LR+ %.3f  exact AUC %.4f" % (tpr, fpr, tpr / fpr, norm.cdf(d / np.sqrt(2))))
for pi in (0.5, 0.05, 0.01):
    prec = pi * tpr / (pi * tpr + (1 - pi) * fpr)
    s, y = make_screen(400000, pi, d, 1)
    print(f"pi {pi:5.2f}: precision at t=1.5 {prec:.4f}   simulated AUC {M.roc_auc_score(y, s):.4f}"
          f"  AP {M.average_precision_score(y, s):.4f}  chance AP {pi}")

print("\n== Q3  Example 57.5-3, three averages")
C = np.array([[20, 2, 0], [3, 4, 3], [1, 1, 6]])
yt = np.repeat([0, 1, 2], C.sum(1)); yp = np.concatenate([np.repeat([0, 1, 2], C[k]) for k in range(3)])
print("per-class F1:", M.f1_score(yt, yp, average=None).round(4), " recall:", M.recall_score(yt, yp, average=None).round(4),
      " precision:", M.precision_score(yt, yp, average=None).round(4), " support", C.sum(1))
for a in ("micro", "macro", "weighted"): print(a, round(M.f1_score(yt, yp, average=a), 4))
print("accuracy", round(M.accuracy_score(yt, yp), 4))
# make class B rarer: 5 true B of 40 with 1 found
C2 = np.array([[24, 1, 1], [3, 1, 1], [1, 0, 8]]); print("second matrix", C2.tolist(), "n", C2.sum())
yt2 = np.repeat([0, 1, 2], C2.sum(1)); yp2 = np.concatenate([np.repeat([0, 1, 2], C2[k]) for k in range(3)])
for a in ("micro", "macro", "weighted"): print(a, round(M.f1_score(yt2, yp2, average=a), 4))
print("class B: F1 %.4f recall %.4f" % (M.f1_score(yt2, yp2, average=None)[1], M.recall_score(yt2, yp2, average=None)[1]))

print("\n== Q4  log loss against AUC (breast cancer, malignant = 1)")
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y); ytr, yte = 1 - ytr, 1 - yte
lr = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xtr, ytr); p_lr = lr.predict_proba(Xte)[:, 1]
nb = GaussianNB().fit(Xtr, ytr); p_nb = nb.predict_proba(Xte)[:, 1]
pr = DummyClassifier(strategy="prior").fit(Xtr, ytr); p_pr = pr.predict_proba(Xte)[:, 1]
for nm, p in (("prior", p_pr), ("naive Bayes", p_nb), ("logistic", p_lr)):
    print(f"{nm:12s} log loss {M.log_loss(yte, p):.4f}  AUC {M.roc_auc_score(yte, p):.4f}  accuracy {M.accuracy_score(yte, p > .5):.4f}")
z = lr.decision_function(Xte)
sig = lambda u: 1 / (1 + np.exp(-u))
print("logistic scores, logit multiplied by a factor (an increasing transformation, AUC is unchanged):")
for k in (0.2, 0.5, 1, 2, 5):
    p = sig(k * z)
    print(f"  factor {k:3.1f}: log loss {M.log_loss(yte, p):.4f}  AUC {M.roc_auc_score(yte, p):.4f}  Brier {M.brier_score_loss(yte, p):.4f}")
zn = np.log(p_nb + 1e-300) - np.log(1 - p_nb + 1e-300)
print("naive Bayes logit range %.1f to %.1f; its probabilities below 0.001 or above 0.999: %d of %d"
      % (np.clip(zn, -700, 700).min(), np.clip(zn, -700, 700).max(), ((p_nb < .001) | (p_nb > .999)).sum(), len(yte)))
print("test prevalence", yte.mean().round(4))
