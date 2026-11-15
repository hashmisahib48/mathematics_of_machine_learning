# Wine: PCA with and without standardising the features.
import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

wine = load_wine()
X, y, names = wine.data, wine.target, np.array(wine.feature_names)
v = X.var(axis=0)
print("variances: largest", v.max().round(1), names[v.argmax()],
      "; smallest", v.min().round(4))

runs = (("raw", [PCA(2)]),
        ("standardised", [StandardScaler(), PCA(2)]))
for label, steps in runs:
    pipe = make_pipeline(*steps).fit(X)
    pca = pipe[-1]
    top = np.argsort(-np.abs(pca.components_), axis=1)[:, :2]
    print(f"\n{label}: variance ratio",
          pca.explained_variance_ratio_.round(3))
    for j in range(2):
        t = ", ".join(f"{names[i]} {pca.components_[j, i]:+.2f}"
                      for i in top[j])
        print(f"  PC{j + 1}: {t}")
    knn = make_pipeline(*steps, KNeighborsClassifier(5))
    acc = cross_val_score(knn, X, y, cv=5).mean()
    print(f"  5-NN on the two scores, 5-fold CV: {acc:.4f}")

pca = make_pipeline(StandardScaler(), PCA()).fit(X)[-1]
cum = np.cumsum(pca.explained_variance_ratio_)
k80 = int(np.searchsorted(cum, 0.8, "right")) + 1
print("\nstandardised: cumulative variance", cum[:4].round(3))
print("components for 80% of the variance:", k80)
