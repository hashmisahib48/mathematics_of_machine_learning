# Depth of a tree for the iris flowers: training and validation.
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
for d in (1, 2, 3, 4, 6, None):
    m = DecisionTreeClassifier(max_depth=d, random_state=0)
    r = cross_validate(m, X, y, cv=5, return_train_score=True)
    print(f"max_depth {str(d):4s}: train {r['train_score'].mean():.3f}"
          f"  validation {r['test_score'].mean():.3f}")
