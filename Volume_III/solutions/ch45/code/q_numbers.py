# Numbers quoted in the model answers to the questions for discussion (Q1 - Q4) of Chapter 45
import sys, numpy as np
sys.path.insert(0, '../../../ch45/code')
from spectra import make_spectra
from scipy.optimize import brentq
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.random_projection import GaussianRandomProjection, johnson_lindenstrauss_min_dim as jl
np.set_printoptions(precision=4, suppress=True)
f = lambda e: e**2 / 2 - e**3 / 3

print("== Q1  what the lemma promises")
print("k = 6000, n = 1000: smallest eps with jl(n, eps) <= 6000 : %.4f" % brentq(lambda e: jl(1000, eps=e) - 6000, 0.05, 0.5))
print("jl(1000, eps=0.1) =", int(jl(1000, eps=0.1)), "  jl(1000, eps=0.5) =", int(jl(1000, eps=0.5)))
print("p = 1e6: k/p = %.4f" % (6000 / 1e6))
X, y = make_spectra(); Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
n, p = Xtr.shape
sv = np.linalg.svd(Xtr - Xtr.mean(0), compute_uv=False)**2
print("spectra: share of the variance in the first 3, 5, 10 principal components: %.3f %.3f %.3f" % tuple(sv[:m].sum() / sv.sum() for m in (3, 5, 10)))
G = Xtr @ Xtr.T; sq = np.diag(G); D2 = sq[:, None] + sq[None] - 2 * G
iu = np.triu_indices(n, 1); same = (ytr[:, None] == ytr[None])[iu]; d2 = D2[iu]
print("pairs: %d;  mean squared distance within a group %.0f, between groups %.0f (ratio %.2f)" % (len(d2), d2[same].mean(), d2[~same].mean(), d2[~same].mean() / d2[same].mean()))
print("   k  sd of q (theory)  max|q-1| over all pairs  share of pairs with |q-1|>0.5  5-NN acc  between/within ratio after")
for k in (10, 30, 100, 317, 735):
    Z = GaussianRandomProjection(k, random_state=0).fit(Xtr).transform(Xtr)
    g = Z @ Z.T; s = np.diag(g); d2p = (s[:, None] + s[None] - 2 * g)[iu]; q = d2p / d2
    Zt = GaussianRandomProjection(k, random_state=0).fit(Xtr).transform(Xte)
    acc = KNeighborsClassifier(5).fit(Z, ytr).score(Zt, yte)
    print("%4d  %12.3f  %16.3f  %22.4f  %17.3f  %10.2f" % (k, np.sqrt(2 / k), np.abs(q - 1).max(), (np.abs(q - 1) > 0.5).mean(), acc, d2p[~same].mean() / d2p[same].mean()))
print("reconstruction by the pseudo-inverse, relative error |x - xhat| / |x| (mean over the 250 test spectra):")
for k in (10, 30, 100, 1000):
    g = GaussianRandomProjection(k, random_state=0, compute_inverse_components=True).fit(Xtr); Xh = g.inverse_transform(g.transform(Xte))
    print("   k = %4d: %.3f   (sqrt(1 - k/p) = %.3f)" % (k, (np.linalg.norm(Xte - Xh, axis=1) / np.linalg.norm(Xte, axis=1)).mean(), np.sqrt(1 - k / p)))

print("\n== Q2  RBFSampler against Nystroem on the digits, D = m = 300")
Dg, Dy = load_digits(return_X_y=True); Dtr, Dte, dtr, dte = train_test_split(Dg, Dy, test_size=0.25, random_state=0, stratify=Dy); Xs = Dtr / 16
print("gamma   offdiag mean K  eigenvalues for 90%% of trace  RBFSampler  Nystroem  best rank-300  max entry error (RBF)  1/sqrt(D)")
for gm in (0.001, 0.1105, 1.0):
    K = rbf_kernel(Xs, gamma=gm); lam = np.linalg.eigvalsh(K)[::-1]; cs = np.cumsum(lam) / lam.sum()
    Zr = RBFSampler(gamma=gm, n_components=300, random_state=0).fit_transform(Xs); Zn = Nystroem(gamma=gm, n_components=300, random_state=0).fit_transform(Xs)
    er = lambda Z: np.linalg.norm(K - Z @ Z.T) / np.linalg.norm(K)
    best = np.sqrt((lam[300:]**2).sum() / (lam**2).sum())
    print("%6.4f  %12.4f  %22d  %14.4f  %10.4f  %12.4f  %20.4f  %9.4f" % (gm, (K.sum() - np.trace(K)) / (len(K)**2 - len(K)), np.searchsorted(cs, 0.9) + 1, er(Zr), er(Zn), best, np.abs(K - Zr @ Zr.T).max(), 1 / np.sqrt(300)))

print("\n== Q3  leakage")
gm = 0.1105
Dte_s = Dte / 16
res = {'train only': [], 'train + test': []}
for seed in range(10):
    pl = make_pipeline(Nystroem(gamma=gm, n_components=100, random_state=seed), LinearSVC(C=1, max_iter=20000)).fit(Xs, dtr); res['train only'].append(pl.score(Dte_s, dte))
    ny = Nystroem(gamma=gm, n_components=100, random_state=seed).fit(np.vstack([Xs, Dte_s]))     # landmarks may come from the test images
    res['train + test'].append(LinearSVC(C=1, max_iter=20000).fit(ny.transform(Xs), dtr).score(ny.transform(Dte_s), dte))
    if seed == 0: print("landmarks that are test images (seed 0, fitted on all 1797): %d of 100" % (ny.component_indices_ >= len(Xs)).sum())
for k_, v in res.items(): print("  Nystroem landmarks from %-12s: accuracy mean %.4f sd %.4f over 10 seeds" % (k_, np.mean(v), np.std(v)))
print("  fraction of the test images that could be a landmark when the pool is all data: %.3f" % (len(Dte_s) / (len(Xs) + len(Dte_s))))
ny = Nystroem(gamma=gm, n_components=100, random_state=0).fit(np.vstack([Xs, Dte_s])); allX = np.vstack([Xs, Dte_s]); Kall = rbf_kernel(allX, gamma=gm); Za = ny.transform(allX)
rowerr = np.linalg.norm(Kall - Za @ Za.T, axis=1) / np.linalg.norm(Kall, axis=1)
isl = np.zeros(len(allX), bool); isl[ny.component_indices_] = True
print("  relative row error of K-hat: landmarks %.2e, other images %.3f" % (rowerr[isl].mean(), rowerr[~isl].mean()))

print("\n== Q4  cost of a prediction")
svc = SVC(kernel='rbf', gamma=gm, C=1).fit(Xs, dtr); nsv = int(svc.n_support_.sum()); pdim = Xs.shape[1]
print("digits: exact SVC (gamma %.4f, C 1): %d support vectors of %d samples, accuracy %.4f" % (gm, nsv, len(Xs), svc.score(Dte_s, dte)))
for m in (50, 200, 800):
    pl = make_pipeline(Nystroem(gamma=gm, n_components=m, random_state=0), LinearSVC(C=1, max_iter=20000)).fit(Xs, dtr)
    print("  Nystroem m = %3d: accuracy %.4f" % (m, pl.score(Dte_s, dte)))
# folding W^-1/2 into the weights: the prediction then needs m kernel values and m products
ny = Nystroem(gamma=gm, n_components=200, random_state=0).fit(Xs); lin = LinearSVC(C=1, max_iter=20000).fit(ny.transform(Xs), dtr)
Cm = rbf_kernel(Dte_s, ny.components_, gamma=gm); Wih = ny.normalization_
dec1 = ny.transform(Dte_s) @ lin.coef_.T + lin.intercept_; dec2 = Cm @ (Wih.T @ lin.coef_.T) + lin.intercept_
print("folded weights give the same decision values:", np.allclose(dec1, dec2))
for nm, n_sv_, p_, m_, Dd in (("digits", nsv, pdim, 200, 800), ("synthetic of §45.9", 7598, 20, 300, 1000)):
    ex = n_sv_ * (p_ + 1); ny_plain = m_ * (p_ + m_ + 1); ny_fold = m_ * (p_ + 1); rf = Dd * (p_ + 1)
    print("%-20s multiplications per prediction: exact %d,  Nystroem as sklearn computes it %d (%.1fx), Nystroem folded %d (%.1fx), RBFSampler D=%d: %d (%.1fx)" % (nm, ex, ny_plain, ex / ny_plain, ny_fold, ex / ny_fold, Dd, rf, ex / rf))
