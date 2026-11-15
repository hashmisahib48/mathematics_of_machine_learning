# SparsePCA on the standardised wine data: sparsity against fit.
import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA, SparsePCA, MiniBatchSparsePCA
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
Xs = StandardScaler().fit_transform(X)        # 178 x 13, centred
total = (Xs ** 2).sum()


def report(name, est):
    Z, C = est.transform(Xs), est.components_
    nz = (np.abs(C) > 1e-12).sum(axis=1)      # non-zero loadings
    rest = ((Xs - Z @ C) ** 2).sum()
    off = np.abs(np.corrcoef(Z.T) - np.eye(3)).max()
    print(f"{name:19s} non-zeros {nz} explained "
          f"{1 - rest / total:.3f} corr {off:.2f}")


report("PCA", PCA(3).fit(Xs))
for a in (0.5, 2, 5):
    report(f"SparsePCA alpha={a}",
           SparsePCA(3, alpha=a, random_state=0).fit(Xs))
mb = MiniBatchSparsePCA(3, alpha=5, batch_size=10, max_iter=200,
                        random_state=0)
report("MiniBatch alpha=5", mb.fit(Xs))
