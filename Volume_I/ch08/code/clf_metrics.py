# 8.4  Confusion matrix and scores for ten samples, then imbalance.
import numpy as np
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])    # true classes
yh = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0])   # predictions
tp = np.sum((y == 1) & (yh == 1))
fn = np.sum((y == 1) & (yh == 0))
fp = np.sum((y == 0) & (yh == 1))
tn = np.sum((y == 0) & (yh == 0))
print("TP FN FP TN:", tp, fn, fp, tn)
print(confusion_matrix(y, yh))   # rows true, columns predicted
prec, rec = tp / (tp + fp), tp / (tp + fn)       # Eq. 8.4-2
f1 = 2 * prec * rec / (prec + rec)               # Eq. 8.4-3
print("accuracy", (tp + tn) / 10,                # Eq. 8.4-1
      " precision", prec, " recall", rec, " F1", round(f1, 4))
print("sklearn:", precision_score(y, yh), recall_score(y, yh),
      round(f1_score(y, yh), 4))
print("recall_score(yh, y), swapped:", recall_score(yh, y))

X, y = load_digits(return_X_y=True)
y = (y == 9).astype(int)               # nine or not
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
for m in (DummyClassifier(), make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000))):
    p = m.fit(X_tr, y_tr).predict(X_te)
    print(confusion_matrix(y_te, p).tolist(),
          " accuracy", round(m.score(X_te, y_te), 3),
          " precision", round(precision_score(
              y_te, p, zero_division=0), 3),
          " recall", round(recall_score(y_te, p), 3))
