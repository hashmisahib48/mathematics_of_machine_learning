# Leave-one-out shortcut, confusion-matrix scores, softmax and log
# loss (Examples R.6-1 and R.8-1).
import numpy as np
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, f1_score, log_loss

x = np.arange(1., 6)
y = np.array([3., 4, 6, 7, 10])
X = np.c_[np.ones(5), x]
H = X @ np.linalg.inv(X.T @ X) @ X.T         # hat matrix
r = y - H @ y
e = r / (1 - np.diag(H))                     # Eq. 10.9-4 (Volume I)
print("h_ii:", np.diag(H).round(2))
print("LOO errors:", e.round(4) + 0.0)
loo = []                                     # refit five times
for i in range(5):
    keep = np.arange(5) != i
    w = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
    loo.append(y[i] - X[i] @ w)
print("by refitting:", np.round(loo, 4) + 0.0)
print("LOO MSE:", round(np.mean(np.square(e)), 4))

yt = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
yp = [1, 1, 1, 1, 0, 1, 1, 0, 0, 0]
tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
print("TP FN FP TN:", tp, fn, fp, tn)
print("accuracy", (tp + tn) / 10, " precision",
      round(tp / (tp + fp), 4), " recall", tp / (tp + fn))
print("F1:", round(2 * tp / (2 * tp + fp + fn), 4),
      "  sklearn:", round(f1_score(yt, yp), 4))

f = np.array([2., 1., 0.])                   # three class scores
P = softmax(f)                               # Eq. 7.5-1
Y = np.array([0., 1., 0.])                   # true class: second
print("P:", P.round(4), " log loss:", round(-np.log(P[1]), 4))
print("gradient P - Y:", (P - Y).round(4))
print("sklearn log_loss:",
      round(log_loss([1], [P], labels=[0, 1, 2]), 4))
