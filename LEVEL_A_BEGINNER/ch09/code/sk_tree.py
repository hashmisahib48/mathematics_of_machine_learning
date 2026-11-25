# The examples of Section 9.4 with scikit-learn.
from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                          export_text)

X = [[1, 4], [2, 7], [3, 2], [5, 6], [6, 2], [6, 8], [7, 5], [8, 4]]
y = [0, 0, 0, 1, 0, 1, 1, 0]
t = DecisionTreeClassifier(random_state=0).fit(X, y)
print(export_text(t, feature_names=["x1", "x2"]))
print("depth", t.get_depth(), " leaves", t.get_n_leaves())
print("importances", t.feature_importances_.round(4))
r = DecisionTreeRegressor(max_depth=1).fit([[1], [2], [3], [4], [5],
                                            [6]], [5, 6, 7, 15, 16, 18])
print("regression leaves", r.tree_.value.ravel()[1:].round(4))
