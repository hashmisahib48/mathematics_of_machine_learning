# Section 57.5, Example 57.5-3: averaging over three classes.
import numpy as np
from sklearn import metrics as m

C = np.array([[20, 2, 0],        # rows: true, columns: predicted
              [3, 4, 3],
              [1, 1, 6]])
y = np.repeat([0, 1, 2], C.sum(axis=1))
p = np.concatenate([np.repeat([0, 1, 2], row) for row in C])
tp, pred, true = np.diag(C), C.sum(axis=0), C.sum(axis=1)
P, R, F = tp / pred, tp / true, 2 * tp / (pred + true)
print("precision", P.round(4), "recall", R.round(4))
print("F1       ", F.round(4), " support", true)
by_hand = {"micro": 2 * tp.sum() / (pred.sum() + true.sum()),
           "macro": F.mean(), "weighted": (F * true).sum() / true.sum()}
for avg, v in by_hand.items():
    assert abs(v - m.f1_score(y, p, average=avg)) < 1e-12
    print(f"{avg:9s}F1 {v:.4f}")
fm = 2 * P.mean() * R.mean() / (P.mean() + R.mean())
print("macro P, R:", P.mean().round(4), R.mean().round(4),
      " F1 of the means:", fm.round(4))
