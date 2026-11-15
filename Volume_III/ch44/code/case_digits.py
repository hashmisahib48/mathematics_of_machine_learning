# Digits in two dimensions: PCA against the manifold methods.
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import (ClassicalMDS, Isomap, MDS, TSNE,
    LocallyLinearEmbedding, SpectralEmbedding, trustworthiness)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
knn = KNeighborsClassifier(n_neighbors=5)
# Chapter 42: PCA(2) refitted inside every fold of a pipeline
pipe = make_pipeline(PCA(2), knn)
print("Chapter 42 pipeline: 5-fold",
      round(cross_val_score(pipe, Xtr, ytr, cv=5).mean(), 4),
      " test", round(pipe.fit(Xtr, ytr).score(Xte, yte), 4))
cmds = ClassicalMDS(2).fit(Xtr)
Zp = PCA(2).fit_transform(Xtr)
sg = np.sign((Zp * cmds.embedding_).sum(axis=0))   # signs differ
print("ClassicalMDS vs PCA scores: max |diff| =",
      f"{np.abs(Zp - sg * cmds.embedding_).max():.1e}")
models = {
    "PCA": PCA(2),
    "MDS (SMACOF)": MDS(2, init="classical_mds", n_init=1,
        metric="euclidean", normalized_stress=True, random_state=0),
    "Isomap": Isomap(n_neighbors=12),
    "LLE": LocallyLinearEmbedding(n_neighbors=8, reg=0.1,
                                  random_state=0),
    "Laplacian": SpectralEmbedding(n_neighbors=10, random_state=0),
    "t-SNE": TSNE(perplexity=30, init="pca", random_state=0),
}
emb = {}
print(f"{'method':14s} {'T(10)':>7s} {'5-fold':>7s} {'test':>7s}")
for name, est in models.items():
    Z = emb[name] = est.fit_transform(Xtr)
    tw = trustworthiness(Xtr, Z, n_neighbors=10)
    cv = cross_val_score(knn, Z, ytr, cv=5).mean()
    te = "-"
    if hasattr(est, "transform"):          # new images possible
        te = f"{knn.fit(Z, ytr).score(est.transform(Xte), yte):.4f}"
    print(f"{name:14s} {tw:7.4f} {cv:7.4f} {te:>7s}")
np.savez("verify/digits_embeddings.npz", **emb)
