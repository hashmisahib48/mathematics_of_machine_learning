# Choosing k for the iris flowers by cross-validation.
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

X, y = load_iris(return_X_y=True)
for k in (1, 3, 5, 9, 15, 25, 50, 100):
    m = KNeighborsClassifier(n_neighbors=k)
    acc = cross_val_score(m, X, y, cv=5)      # five train/test splits
    print(f"k = {k:3d}: mean accuracy {acc.mean():.3f}")
