# Problem 16B.2: class_weight="balanced" on imbalanced data.
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=2000, n_features=6,
    n_informative=3, n_redundant=0, weights=[0.9, 0.1],
    class_sep=1.0, flip_y=0.02, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.5, random_state=0, stratify=y)
print("share of class 1: train", y_tr.mean(), " test", y_te.mean())

n, counts = len(y_tr), np.bincount(y_tr)
w = n / (2 * counts)                           # the "balanced" rule
print("class counts", counts, " weights n/(2 n_k):", np.round(w, 3),
      " ratio", round(float(w[1] / w[0]), 3))
for cw in (None, "balanced"):
    m = LogisticRegression(class_weight=cw).fit(X_tr, y_tr)
    pred = m.predict(X_te)
    tp = int(((pred == 1) & (y_te == 1)).sum())
    print(f"class_weight={cw}: recall {recall_score(y_te, pred):.4f}"
          f"  precision {precision_score(y_te, pred):.4f}"
          f"  accuracy {m.score(X_te, y_te):.4f}")
    print(f"   predicted positives {pred.sum()} (true {tp}), "
          f"actual positives {y_te.sum()}, "
          f"intercept {m.intercept_[0]:.3f}")
    print("   mean probability of class 1:",
          round(float(m.predict_proba(X_te)[:, 1].mean()), 3))
print("ln of the weight ratio:", round(float(np.log(w[1] / w[0])), 3))
