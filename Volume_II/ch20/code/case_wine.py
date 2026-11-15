# Case study: three kinds of wine, 13 chemical measurements.
import numpy as np
from sklearn.datasets import load_wine
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
    GridSearchCV, cross_val_score, train_test_split)
from sklearn.neighbors import (
    KNeighborsClassifier, NearestCentroid,
    NeighborhoodComponentsAnalysis as NCA)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
print("train", Xtr.shape, "test", Xte.shape, np.bincount(y))
sd = Xtr.std(axis=0)
print(f"feature s.d. {sd.min():.2f} to {sd.max():.1f}")

def report(name, model):
    model.fit(Xtr, ytr)
    cv = cross_val_score(model, Xtr, ytr, cv=5).mean()
    tr, te = model.score(Xtr, ytr), model.score(Xte, yte)
    print(f"{name:25s} train {tr:.4f}  test {te:.4f}  cv {cv:.4f}")

sc = StandardScaler
report("majority class", DummyClassifier())
report("5-NN, raw", KNeighborsClassifier())
report("5-NN, scaled", make_pipeline(sc(), KNeighborsClassifier()))
knn_d = KNeighborsClassifier(weights="distance")
report("5-NN 1/d, scaled", make_pipeline(sc(), knn_d))
report("nearest centroid, raw", NearestCentroid())
report("nearest centroid, scaled",
       make_pipeline(sc(), NearestCentroid()))
nca = NCA(n_components=2, random_state=0)
report("NCA (2) + 5-NN, scaled",
       make_pipeline(sc(), nca, KNeighborsClassifier()))

name = "kneighborsclassifier__n_neighbors"
grid = {name: range(1, 26)}
pipe = make_pipeline(sc(), KNeighborsClassifier())
best = GridSearchCV(pipe, grid, cv=5).fit(Xtr, ytr)
k = best.best_params_[name]
print(f"k by cross-validation: {k}, cv {best.best_score_:.4f},"
      f" test {best.score(Xte, yte):.4f}")
