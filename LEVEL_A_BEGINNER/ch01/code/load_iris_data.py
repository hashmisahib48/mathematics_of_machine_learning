# Inspect a bundled data set: the iris flowers.
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
print("shape of X:", X.shape, " shape of y:", y.shape)
print("features:", iris.feature_names[:2])
print("          ", iris.feature_names[2:])
print("classes: ", iris.target_names.tolist())
for i in (0, 50, 100):                         # one flower of each kind
    print(i, X[i], "->", iris.target_names[y[i]])
