# Problem 25B.1: the out-of-bag score of BaggingClassifier by hand
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
n = len(y_tr)
bag = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=50, oob_score=True,
    random_state=0).fit(X_tr, y_tr)

# Eq. 25.5-2: add the class fractions of the trees that missed i
total = np.zeros((n, 2))
votes = np.zeros(n, int)           # OOB trees per sample
for tree, idx in zip(bag.estimators_, bag.estimators_samples_):
    out = np.ones(n, bool)
    out[idx] = False               # idx: the drawn samples
    total[out] += tree.predict_proba(X_tr[out])
    votes[out] += 1
has_oob = votes > 0
pred = total.argmax(axis=1)        # 1/|B_i| drops out
manual = (pred[has_oob] == y_tr[has_oob]).mean()   # Eq. 25.5-3

print("oob_score_ of scikit-learn: %.4f" % bag.oob_score_)
print("recomputed by hand:         %.4f" % manual)
print("equal to the last digit:", np.isclose(manual, bag.oob_score_))
print("samples without an out-of-bag tree:", int((~has_oob).sum()),
      " (expected %.1e)" % (n * (1 - np.exp(-1)) ** 50))
print("out-of-bag trees per sample: mean %.2f (0.368 * 50 = %.2f), "
      "min %d" % (votes.mean(), 0.368 * 50, votes.min()))
distinct = [len(np.unique(i)) for i in bag.estimators_samples_]
print("distinct samples per tree: mean %.1f   0.632 n = %.1f   n = %d"
      % (np.mean(distinct), 0.632 * n, n))
