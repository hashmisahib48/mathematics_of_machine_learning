# Swiss roll and S-curve: every method, one table.
import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import make_s_curve, make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import (ClassicalMDS, Isomap, MDS, TSNE,
    LocallyLinearEmbedding, SpectralEmbedding, trustworthiness)

def methods():
    return {
        "PCA": PCA(n_components=2),
        "MDS": MDS(n_components=2, init="classical_mds", n_init=1,
                   metric="euclidean", normalized_stress=True,
                   random_state=0),
        "Isomap": Isomap(n_neighbors=10, n_components=2),
        "LLE": LocallyLinearEmbedding(n_neighbors=12,
                                      random_state=0),
        "Hessian LLE": LocallyLinearEmbedding(n_neighbors=12,
            n_components=2, method="hessian", random_state=0),
        "Laplacian": SpectralEmbedding(n_neighbors=10,
                                       random_state=0),
        "t-SNE": TSNE(n_components=2, perplexity=30, init="pca",
                      random_state=0),
    }

out = {}
for name, make in (("swiss roll", make_swiss_roll),
                   ("S-curve", make_s_curve)):
    X, t = make(1000, random_state=0)
    print(name, X.shape)
    print(f"{'method':12s} {'T(10)':>7s} {'|rho(z,t)|':>11s}")
    for m, est in methods().items():
        Z = est.fit_transform(X)
        T = trustworthiness(X, Z, n_neighbors=10)
        rho = max(abs(spearmanr(Z[:, j], t)[0]) for j in (0, 1))
        print(f"{m:12s} {T:7.4f} {rho:11.3f}")
        out[f"{name}|{m}"] = Z
np.savez("verify/swiss_embeddings.npz", **out)
