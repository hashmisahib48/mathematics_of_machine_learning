# Missing values in a tree: where does NaN go?
import numpy as np
from sklearn.tree import DecisionTreeClassifier

nan = np.nan
X = np.array([[1], [2], [3], [4], [5], [6], [nan], [nan]])
y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
with_nan = DecisionTreeClassifier(random_state=0).fit(X, y)
print("NaN seen in training: missing_go_to_left =",
      with_nan.tree_.missing_go_to_left[0], " threshold",
      with_nan.tree_.threshold[0])
print("  predict NaN ->", with_nan.predict([[nan]]))

X2 = X[:6]                     # no NaN; larger side: left
y2 = np.array([0, 0, 0, 0, 1, 1])
no_nan = DecisionTreeClassifier(random_state=0).fit(X2, y2)
t = no_nan.tree_
print("no NaN in training: left child n =", t.n_node_samples[1],
      " right child n =", t.n_node_samples[2],
      " missing_go_to_left =", t.missing_go_to_left[0])
print("  predict NaN ->", no_nan.predict([[nan]]))
