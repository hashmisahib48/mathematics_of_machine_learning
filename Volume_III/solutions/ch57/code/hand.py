# Hand checks for Problems 57A.1 - 57A.3 (every number quoted in the solutions)
import sys
import numpy as np
from sklearn import metrics as M

print("== 57A.1  TP = 30, FN = 10, FP = 20, TN = 940")
TP, FN, FP, TN = 30, 10, 20, 940
n = TP + FN + FP + TN
acc = (TP + TN) / n; prec = TP / (TP + FP); rec = TP / (TP + FN)
F1 = 2 * TP / (2 * TP + FN + FP)
F2 = 5 * TP / (5 * TP + 4 * FN + FP)
num = TP * TN - FP * FN
den = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
mcc = num / den
print("n =", n, " accuracy", acc, " precision", prec, " recall", rec)
print("F1 = 2TP/(2TP+FN+FP) = %d/%d = %.4f" % (2 * TP, 2 * TP + FN + FP, F1))
print("F2 = 5TP/(5TP+4FN+FP) = %d/%d = %.4f" % (5 * TP, 5 * TP + 4 * FN + FP, F2))
print("harmonic mean of P and R:", round(2 * prec * rec / (prec + rec), 4),
      " F2 from P,R:", round(5 * prec * rec / (4 * prec + rec), 4))
print("MCC numerator", num, " (TP+FP)(TP+FN)(TN+FP)(TN+FN) =",
      (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN), " sqrt = %.2f" % den,
      " MCC = %.4f" % mcc)
y = np.r_[np.ones(TP + FN), np.zeros(FP + TN)].astype(int)
p = np.r_[np.ones(TP), np.zeros(FN), np.ones(FP), np.zeros(TN)].astype(int)
print("sklearn:", round(M.accuracy_score(y, p), 4), round(M.precision_score(y, p), 4),
      round(M.recall_score(y, p), 4), round(M.f1_score(y, p), 4),
      round(M.fbeta_score(y, p, beta=2), 4), round(M.matthews_corrcoef(y, p), 4))
print("prevalence", (TP + FN) / n, " nobody-is-ill accuracy", (FP + TN) / n,
      " balanced accuracy", round((rec + TN / (TN + FP)) / 2, 4))
pe = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / n ** 2
print("kappa: pe %.4f, kappa %.4f (sklearn %.4f)" % (pe, (acc - pe) / (1 - pe), M.cohen_kappa_score(y, p)))

print("\n== 57A.2  labels 1 0 1 0 1 0, scores .8 .7 .6 .4 .3 .2")
yy = np.array([1, 0, 1, 0, 1, 0]); s = np.array([.8, .7, .6, .4, .3, .2])
pos, neg = s[yy == 1], s[yy == 0]
wins = sum(a > b for a in pos for b in neg); ties = sum(a == b for a in pos for b in neg)
print("positive scores", pos.tolist(), " negative scores", neg.tolist())
for a in pos: print("  ", a, "beats", [float(b) for b in neg if a > b])
print("wins", wins, "ties", ties, "of", len(pos) * len(neg), " AUC =", round((wins + .5 * ties) / 9, 4),
      " sklearn", round(M.roc_auc_score(yy, s), 4))
print("sorted scores and labels:", list(zip(s.tolist(), yy.tolist())))
tp = np.cumsum(yy); fp = np.cumsum(1 - yy); prc = tp / (tp + fp); rc = tp / yy.sum()
for k in range(6): print("  threshold %.1f  TP %d FP %d  precision %.4f  recall %.4f" % (s[k], tp[k], fp[k], prc[k], rc[k]))
steps = np.diff(np.r_[0, rc])
print("recall steps", steps.round(4), " AP = sum step*precision =", round((steps * prc).sum(), 4),
      " sklearn", round(M.average_precision_score(yy, s), 4))
print("precision at the three positives:", prc[yy == 1].round(4), " mean", round(prc[yy == 1].mean(), 4))
f, t, _ = M.roc_curve(yy, s, drop_intermediate=False)
print("ROC points (FPR, TPR):", list(zip(f.round(3).tolist(), t.round(3).tolist())))
print("trapezoid area of the precision-recall points (not AP):",
      round(np.trapezoid(np.r_[1, prc], np.r_[0, rc]), 4))

print("\n== 57A.3  sensitivity 0.90, specificity 0.95")
se, sp = 0.90, 0.95
LR = se / (1 - sp); print("LR+ = %.4f / %.4f = %.1f" % (se, 1 - sp, LR))
for pi in (0.01, 0.10, 0.50):
    odds = pi / (1 - pi) * LR; ppv = odds / (1 + odds)
    direct = se * pi / (se * pi + (1 - sp) * (1 - pi))
    print("pi %.2f: prior odds %.5f  posterior odds %.5f  precision %.4f  (direct %.4f)  false alarms per true alarm %.2f"
          % (pi, pi / (1 - pi), odds, ppv, direct, (1 - ppv) / ppv))
n_ = 10000
for pi in (0.01, 0.10, 0.50):
    ill = n_ * pi; well = n_ - ill
    print("pi %.2f: of %d people, ill %d -> flagged ill %.0f, well %d -> flagged well %.0f" % (pi, n_, ill, se * ill, well, (1 - sp) * well))
print("prevalence at which precision = 0.5:", round(1 / (1 + LR), 4))
