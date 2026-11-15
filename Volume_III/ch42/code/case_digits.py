# Digits: how many principal components does a classifier need?
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
pca = PCA().fit(Xtr)                          # all 64 components
cum = np.cumsum(pca.explained_variance_ratio_)
print("variance kept by 2, 10, 20 components:",
      cum[[1, 9, 19]].round(3))
for frac in (0.80, 0.90, 0.95, 0.99):         # a float n_components
    k = PCA(frac).fit(Xtr).n_components_
    print(f"variance fraction {frac:.2f} -> {k} components")
print("n_components='mle' ->", PCA("mle").fit(Xtr).n_components_)

knn = KNeighborsClassifier(n_neighbors=5)
base = knn.fit(Xtr, ytr).score(Xte, yte)
print(f"\n5-NN on the 64 pixels: test {base:.4f}")
print("  k   5-fold CV   test   whitened test")
for k in (2, 5, 10, 20, 30, 40, 64):
    model = make_pipeline(PCA(k), knn)
    cv = cross_val_score(model, Xtr, ytr, cv=5).mean()
    te = model.fit(Xtr, ytr).score(Xte, yte)
    wh = make_pipeline(PCA(k, whiten=True), knn)
    tw = wh.fit(Xtr, ytr).score(Xte, yte)
    print(f"{k:3d}   {cv:.4f}    {te:.4f}   {tw:.4f}")
