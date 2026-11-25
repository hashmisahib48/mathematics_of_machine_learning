# Split, fit, predict and score a classifier, with a baseline.
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          random_state=1, stratify=y)
base = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
knn = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr)
print("test flowers:", len(y_te))
print("baseline accuracy:", round(base.score(X_te, y_te), 3))
print("k-NN accuracy:    ", round(knn.score(X_te, y_te), 3))
print("k-NN errors:", (knn.predict(X_te) != y_te).sum())
