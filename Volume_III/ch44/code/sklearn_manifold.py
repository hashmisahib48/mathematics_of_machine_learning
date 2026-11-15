# The five scikit-learn manifold estimators on a swiss roll.
import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import (ClassicalMDS, Isomap, MDS, TSNE,
    LocallyLinearEmbedding, SpectralEmbedding, trustworthiness)

X, t = make_swiss_roll(600, random_state=0)
est = {
    "ClassicalMDS": ClassicalMDS(n_components=2),
    "MDS": MDS(n_components=2, init="classical_mds", n_init=1,
               metric="euclidean", normalized_stress=True,
               random_state=0),
    "Isomap": Isomap(n_neighbors=10, n_components=2),
    "LLE": LocallyLinearEmbedding(n_neighbors=12, n_components=2,
               random_state=0),
    "SpectralEmbedding": SpectralEmbedding(n_components=2,
               n_neighbors=10, random_state=0),
    "TSNE": TSNE(n_components=2, perplexity=30, init="pca",
               random_state=0),
}
for name, m in est.items():
    Z = m.fit_transform(X)
    tw = trustworthiness(X, Z, n_neighbors=10)
    extra = {"ClassicalMDS": lambda: m.eigenvalues_.round(1),
        "MDS": lambda: (round(float(m.stress_), 4), m.n_iter_),
        "Isomap": lambda: round(m.reconstruction_error(), 4),
        "LLE": lambda: f"{m.reconstruction_error_:.2e}",
        "SpectralEmbedding": lambda: m.affinity_matrix_.shape,
        "TSNE": lambda: (round(m.kl_divergence_, 3), m.n_iter_,
                         float(m.learning_rate_))}[name]()
    print(f"{name:18s} trust {tw:.4f}  {extra}  "
          f"transform: {hasattr(m, 'transform')}")
