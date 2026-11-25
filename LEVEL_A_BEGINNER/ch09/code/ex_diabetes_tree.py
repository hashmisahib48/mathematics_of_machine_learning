# A regression tree for the diabetes data: limiting the leaf size.
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

X, y = load_diabetes(return_X_y=True)
a, b, c, d = train_test_split(X, y, test_size=0.25, random_state=0)
for leaf in (1, 5, 10, 20, 40, 80):
    m = DecisionTreeRegressor(min_samples_leaf=leaf, random_state=0)
    m.fit(a, c)
    print(f"min_samples_leaf {leaf:2d}: leaves {m.get_n_leaves():3d},"
          f" train R2 {m.score(a, c):.3f}, test R2 {m.score(b, d):.3f}")
