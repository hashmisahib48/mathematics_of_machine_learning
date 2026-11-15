# Section 57.5, Example 57.5-2: ROC and PR curves of 8 scores.
import numpy as np
from sklearn import metrics as m

y = np.array([1, 0, 1, 1, 0, 1, 0, 0])
s = np.array([.9, .8, .7, .6, .6, .4, .3, .2])
pos, neg = s[y == 1], s[y == 0]
wins = (pos[:, None] > neg[None, :]).sum()       # pairs, Eq. 57.3-7
ties = (pos[:, None] == neg[None, :]).sum()
auc = (wins + 0.5 * ties) / (len(pos) * len(neg))
print("wins", wins, "ties", ties, "of", len(pos) * len(neg),
      " AUC =", auc)

t = np.unique(s)[::-1]                    # distinct thresholds
tpr = np.array([(pos >= u).mean() for u in t])
fpr = np.array([(neg >= u).mean() for u in t])
prec = np.array([(pos >= u).sum() / (s >= u).sum() for u in t])
for row in zip(t, fpr, tpr, prec):
    print("t=%.1f  FPR %.2f  TPR %.2f  precision %.4f" % row)
x, v = np.r_[0, fpr], np.r_[0, tpr]
area = np.sum(np.diff(x) * (v[1:] + v[:-1]) / 2)   # trapezoids
ap = np.sum(np.diff(v) * prec)                     # Eq. 57.3-8
pr, rc, _ = m.precision_recall_curve(y, s)
print("trapezoid AUC", area, " AP", round(ap, 4),
      " trapezoid PR area", round(m.auc(rc, pr), 4))
assert abs(area - m.roc_auc_score(y, s)) < 1e-12
assert abs(ap - m.average_precision_score(y, s)) < 1e-12
