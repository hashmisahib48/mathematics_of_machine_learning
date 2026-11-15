# Numbers quoted in the prose of Chapter 42 that no listing prints.
import json
import warnings
import numpy as np
from scipy import sparse
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
from sklearn.decomposition import SparsePCA, dict_learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine

out = {}
X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
p = PCA().fit(Xtr)
ev = p.explained_variance_
out["digits_train_shape"] = list(Xtr.shape)
out["constant_pixels"] = int((Xtr.std(axis=0) == 0).sum())
out["rank_nonzero_eig"] = int((ev > 1e-10).sum())
out["smallest_eig"] = [float(v) for v in ev[-4:]]
out["auto_solver_digits"] = PCA(10).fit(Xtr)._fit_svd_solver
w = PCA(64, whiten=True).fit(Xtr)
out["whiten_min_scale_clipped"] = int(
    (np.sqrt(w.explained_variance_) < np.finfo(float).eps).sum())
Zw = w.transform(Xte)
out["whiten_max_abs_test_score"] = float(np.abs(Zw).max())
out["whiten_ft_equals_t"] = bool(np.allclose(
    PCA(10, whiten=True).fit_transform(Xtr),
    PCA(10, whiten=True).fit(Xtr).transform(Xtr)))

# kernel PCA with the linear kernel equals PCA
kp = KernelPCA(5, kernel="linear").fit(Xtr[:300])
pp = PCA(5).fit(Xtr[:300])
out["kpca_linear_equals_pca"] = bool(np.allclose(
    np.abs(kp.transform(Xtr[:300])), np.abs(pp.transform(Xtr[:300])),
    atol=1e-6))
out["kpca_eig_over_n_minus_1_equals_pca_var"] = bool(np.allclose(
    kp.eigenvalues_ / 299, pp.explained_variance_, atol=1e-6))

# error messages
def msg(f):
    try:
        f()
    except Exception as e:
        return type(e).__name__ + ": " + str(e)[:110].replace("\n", " ")
    return "no error"
Xn = Xtr[:50].copy(); Xn[0, 0] = np.nan
out["nan"] = msg(lambda: PCA(2).fit(Xn))
out["sparse_randomized"] = msg(lambda: PCA(
    2, svd_solver="randomized").fit(sparse.csr_matrix(Xtr)))
out["float_randomized"] = msg(lambda: PCA(
    0.9, svd_solver="randomized").fit(Xtr))
out["too_many"] = msg(lambda: PCA(70).fit(Xtr))
out["mle_wide"] = msg(lambda: PCA("mle").fit(Xtr[:30]))
out["arpack_all"] = msg(lambda: PCA(
    64, svd_solver="arpack").fit(Xtr))
ip = IncrementalPCA(10).fit(Xtr)
out["inc_batch_size_"] = int(ip.batch_size_)

# sparse PCA objective
Xs = StandardScaler().fit_transform(load_wine().data)
sp = SparsePCA(3, alpha=5, random_state=0).fit(Xs)
code, dic, E, it = dict_learning(
    Xs.T, 3, alpha=5, tol=1e-8, max_iter=1000, method="lars",
    random_state=0, return_n_iter=True)
obj = 0.5 * ((Xs.T - code @ dic) ** 2).sum() + 5 * np.abs(code).sum()
out["sparse_objective"] = float(obj)
out["sparse_error_last"] = float(sp.error_[-1])
out["sparse_nz"] = (np.abs(sp.components_) > 1e-12).sum(axis=1).tolist()
out["sparse_n_iter"] = int(sp.n_iter_)
print(json.dumps(out, indent=1))
json.dump(out, open("verify/prose_numbers.json", "w"), indent=1)
