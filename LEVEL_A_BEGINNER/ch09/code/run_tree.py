# The worked examples of Section 9.4.
from tree import DecisionTreeClassifier, DecisionTreeRegressor, gini

X = [[1, 4], [2, 7], [3, 2], [5, 6], [6, 2], [6, 8], [7, 5], [8, 4]]
y = [0, 0, 0, 1, 0, 1, 1, 0]            # points A to H
t = DecisionTreeClassifier().fit(X, y)
print("root Gini", gini(y))
print("best root split (Gini, feature, threshold):",
      t._best_split(X, y))
r = t.tree_["right"]
print("right child split: x%d <= %.1f" % (r["feature"] + 1,
                                          r["threshold"]))
print("predictions for (4, 7), (7, 3):", t.predict([[4, 7], [7, 3]]))
x = [[1], [2], [3], [4], [5], [6]]
v = [5, 6, 7, 15, 16, 18]
g = DecisionTreeRegressor(max_depth=1).fit(x, v)
leaves = [round(g.tree_[s]["value"], 4) for s in ("left", "right")]
print("regression split x <=", g.tree_["threshold"], " leaves", leaves)
