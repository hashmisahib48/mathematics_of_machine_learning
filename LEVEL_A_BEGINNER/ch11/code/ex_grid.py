# Tuning k-NN for the breast cancer data by grid search.
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
a, b, c, d = train_test_split(X, y, test_size=0.25,
                              random_state=0, stratify=y)
pipe = Pipeline([("scale", StandardScaler()),
                 ("knn", KNeighborsClassifier())])
grid = {"knn__n_neighbors": [1, 3, 5, 7, 9, 15, 25],
        "knn__weights": ["uniform", "distance"]}
g = GridSearchCV(pipe, grid, cv=5, scoring="accuracy").fit(a, c)
print("best parameters", g.best_params_)
print("best cross-validated accuracy", round(g.best_score_, 3))
print("test accuracy of the refitted model", round(g.score(b, d), 3))
