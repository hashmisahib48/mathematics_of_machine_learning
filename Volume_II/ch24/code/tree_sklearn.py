# The loan tree in scikit-learn: fit, read, predict.
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

X = np.array([[35, 22], [27, 40], [22, 55], [48, 28], [33, 62],
              [24, 18], [52, 36], [29, 26], [41, 48], [58, 75]])
y = np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 1])

tree = DecisionTreeClassifier(random_state=0).fit(X, y)
print(export_text(tree, feature_names=["age", "income"]))
t = tree.tree_
print("node  left right feat  thr   n  gini  value (0, 1)")
for k in range(t.node_count):
    print(f"{k:4d} {t.children_left[k]:5d} {t.children_right[k]:5d}"
          f" {t.feature[k]:4d} {t.threshold[k]:5.1f} "
          f"{t.n_node_samples[k]:3d} {t.impurity[k]:5.3f}  "
          f"{np.round(t.value[k, 0], 3)}")
new = np.array([[30, 45], [23, 60]])
print("predict:", tree.predict(new))
stump = DecisionTreeClassifier(max_depth=1, random_state=0)
stump.fit(X, y)
print("stump proba:", stump.predict_proba(new).round(3).tolist())
print("leaf ids:", tree.apply(new), " depth", tree.get_depth(),
      " leaves", tree.get_n_leaves())
print("importances:", tree.feature_importances_.round(4))
