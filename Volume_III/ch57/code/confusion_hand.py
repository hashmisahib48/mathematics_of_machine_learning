# Section 57.5, Example 57.5-1: one confusion matrix, every score.
import textwrap
import numpy as np
from sklearn import metrics as m

TP, FN, FP, TN = 7, 3, 19, 171          # 10 ill, 190 well
y = np.r_[np.ones(TP + FN), np.zeros(FP + TN)].astype(int)
p = np.r_[np.ones(TP), np.zeros(FN), np.ones(FP), np.zeros(TN)]
p = p.astype(int)
n = TP + FN + FP + TN
rec, spec, prec = TP / (TP + FN), TN / (TN + FP), TP / (TP + FP)
F = lambda b: (1 + b**2) * TP / ((1 + b**2) * TP + b**2 * FN + FP)
mcc = (TP * TN - FP * FN) / np.sqrt(
    (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
pe = ((TP + FN) * (TP + FP) + (TN + FP) * (TN + FN)) / n**2
by_hand = {"accuracy": (TP + TN) / n, "recall": rec,
           "specificity": spec, "precision": prec,
           "balanced": (rec + spec) / 2, "F1": F(1), "F2": F(2),
           "F0.5": F(0.5), "MCC": mcc,
           "kappa": ((TP + TN) / n - pe) / (1 - pe),
           "Jaccard": TP / (TP + FN + FP)}
sk = [m.accuracy_score(y, p), m.recall_score(y, p),
      m.recall_score(y, p, pos_label=0), m.precision_score(y, p),
      m.balanced_accuracy_score(y, p), m.f1_score(y, p),
      m.fbeta_score(y, p, beta=2), m.fbeta_score(y, p, beta=0.5),
      m.matthews_corrcoef(y, p), m.cohen_kappa_score(y, p),
      m.jaccard_score(y, p)]
for (k, v), ref in zip(by_hand.items(), sk):
    assert abs(v - ref) < 1e-12
print(textwrap.fill("  ".join(f"{k}={v:.4f}"
                             for k, v in by_hand.items()), 62))
print("p_e =", round(pe, 4), "  LR+ =", round(rec / (1 - spec), 2))
