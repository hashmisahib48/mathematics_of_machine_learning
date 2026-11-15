# Greedy search and the exclusive-or: the root has no gain.
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])                    # exclusive or
names = ["x1", "x2"]
tree = DecisionTreeClassifier(random_state=0).fit(X, y)
print(export_text(tree, feature_names=names))
print("training accuracy:", tree.score(X, y))
t = tree.tree_
a, b = t.children_left[0], t.children_right[0]
print("root impurity:", t.impurity[0], " children:",
      t.impurity[a], t.impurity[b])

stop = DecisionTreeClassifier(min_impurity_decrease=1e-6,
                              random_state=0).fit(X, y)
print("min_impurity_decrease=1e-6: leaves =", stop.get_n_leaves(),
      " accuracy =", stop.score(X, y))
