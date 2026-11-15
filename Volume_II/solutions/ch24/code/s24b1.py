# Problem 24B.1: Listing 24.2 with the entropy of Eq. 24.3-4
import sys
import numpy as np
sys.path.insert(0, "../../../ch24/code")   # book's tree_scratch.py
from tree_scratch import MyTree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DTC


class MyEntropyTree(MyTree):
    def _gini(self, n_k):               # same name: _split is reused
        p = n_k / n_k.sum(axis=-1, keepdims=True)
        logp = np.log2(np.where(p > 0, p, 1.0))   # log 1 = 0
        return -(p * logp).sum(axis=-1)

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
print("root impurity: entropy %.4f, Gini %.4f" % (
    MyEntropyTree()._gini(np.bincount(y_tr)),
    MyTree()._gini(np.bincount(y_tr))))
for d in (1, 2, 3):
    mine = MyEntropyTree(max_depth=d).fit(X_tr, y_tr)
    ref = DTC(criterion="entropy", max_depth=d, random_state=0)
    ref.fit(X_tr, y_tr)
    gin = DTC(max_depth=d, random_state=0).fit(X_tr, y_tr)
    same = np.array_equal(mine.predict(X_te), ref.predict(X_te))
    prob = np.allclose(mine.predict_proba(X_te),
                       ref.predict_proba(X_te))
    print("depth %d: nodes %d and %d, same classes %s, same "
          "fractions %s" % (d, len(mine.nodes_), ref.tree_.node_count,
                            same, prob))
    print("   test accuracy: entropy %.4f, Gini %.4f" % (
        ref.score(X_te, y_te), gin.score(X_te, y_te)))
print("root question: feature %d <= %.2f (entropy), %d <= "
      "%.2f (Gini)" % (ref.tree_.feature[0], ref.tree_.threshold[0],
                       gin.tree_.feature[0], gin.tree_.threshold[0]))
