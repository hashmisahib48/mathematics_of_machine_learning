# Checks behind statements about NMF in the text (Chapter 43)
import json, warnings
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import NMF, MiniBatchNMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

OUT = {}
X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      random_state=0, stratify=y)
# 1. reconstruction_err_ is the Frobenius norm, or sqrt(2 * divergence)
m = NMF(10, max_iter=1000, random_state=0)
W = m.fit_transform(Xtr)
OUT["err_fro_attr"] = m.reconstruction_err_
OUT["err_fro_norm"] = np.linalg.norm(Xtr - W @ m.components_)
Xp = Xtr + 1e-3                                    # KL needs positive data
kl = NMF(10, solver="mu", beta_loss="kullback-leibler", max_iter=1000,
         init="nndsvda", random_state=0)
Wk = kl.fit_transform(Xp); R = Wk @ kl.components_
D = np.sum(Xp * np.log(Xp / R) - Xp + R)
OUT["kl_attr"] = kl.reconstruction_err_
OUT["kl_sqrt2D"] = float(np.sqrt(2 * D))
# 2. regularisation scaling: duplicate the samples -> same H
rng = np.random.RandomState(0)
Xs = Xtr[:300]
W0 = rng.uniform(0.5, 1.5, (300, 6)); H0 = rng.uniform(0.5, 1.5, (6, 64))
kw = dict(init="custom", alpha_W=0.05, l1_ratio=1.0, max_iter=3000,
          tol=1e-10, solver="cd")
a = NMF(6, **kw).fit(Xs, W=W0, H=H0)
b = NMF(6, **kw).fit(np.vstack([Xs, Xs]), W=np.vstack([W0, W0]), H=H0)
OUT["dup_H_maxdiff"] = float(np.abs(a.components_ - b.components_).max())
OUT["dup_H_scale"] = float(np.abs(a.components_).max())
# KKT of the L1-penalised H problem at positive entries:
#   W^T (W H - X) + n_samples * alpha_H * l1_ratio = 0
n = Xs.shape[0]
Wf = NMF(6, **kw).fit_transform(Xs, W=W0, H=H0)
Ha = NMF(6, **kw).fit(Xs, W=W0, H=H0).components_
G = Wf.T @ (Wf @ Ha - Xs) + n * 0.05 * 1.0
pos = Ha > 1e-8
OUT["kkt_max_on_positive"] = float(np.abs(G[pos]).max())
OUT["kkt_min_on_zero"] = float(G[~pos].min()) if (~pos).any() else None
OUT["kkt_G_scale"] = float(n * 0.05)
# 3. multiplicative updates cannot revive a zero
H0z = H0.copy(); H0z[0, :10] = 0.0
mz = NMF(6, solver="mu", init="custom", max_iter=200, tol=0.0).fit(Xs, W=W0, H=H0z)
OUT["mu_zero_stays_zero"] = bool((mz.components_[0, :10] == 0).all())
mc = NMF(6, solver="cd", init="custom", max_iter=200, tol=0.0).fit(Xs, W=W0, H=H0z)
OUT["cd_zero_revives"] = bool((mc.components_[0, :10] > 0).any())
# 4. defaults
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    d = NMF(random_state=0, max_iter=2).fit(Xtr)
OUT["default_n_components_"] = int(d.n_components_)
# 5. messages
for label, f in [("negative", lambda: NMF(3).fit(Xtr - 1)),
                 ("cd_kl", lambda: NMF(3, beta_loss="kullback-leibler").fit(Xtr)),
                 ]:
    try: f()
    except ValueError as e: OUT["msg_" + label] = str(e).splitlines()[0][:120]
with warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter("always")
    NMF(3, solver="mu", init="nndsvd", max_iter=5, random_state=0).fit(Xtr)
OUT["warn_nndsvd_mu"] = [str(w.category.__name__) for w in ws if "zeros" in str(w.message)]
# 6. MiniBatchNMF
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    t = MiniBatchNMF(10, random_state=0, batch_size=256, max_iter=100).fit(Xtr)
OUT["mb_err"] = t.reconstruction_err_; OUT["mb_n_iter"] = int(t.n_iter_)
# 7. L1 penalties on the digits (k = 10): sparsity, error, dead parts
reg = []
for aW, aH in [(0, 0), (0, 0.01), (0, 0.1), (0, 1.0), (0.001, 0.001)]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        r = NMF(10, alpha_W=aW, alpha_H=aH, l1_ratio=1.0, max_iter=1000,
                random_state=0).fit(Xtr)
    Hr = r.components_
    reg.append([aW, aH, round(float(r.reconstruction_err_), 1),
                round(float((Hr == 0).mean()), 3),
                int((Hr > 1e-12).any(axis=1).sum()), int(r.n_iter_)])
OUT["reg_aW_aH_err_Hzero_alive_iter"] = reg
# alpha_W alone: the penalty can be shrunk by rescaling (W/c, cH)
base = NMF(10, max_iter=1000, random_state=0).fit(Xtr)
OUT["base_H_max"] = float(base.components_.max())
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    dr = NMF(10, alpha_W=0.01, l1_ratio=1.0, max_iter=300,
             random_state=0).fit(Xtr)
OUT["aW_only_n_iter"] = int(dr.n_iter_)
OUT["aW_only_H_max"] = float(dr.components_.max())
for k, v in OUT.items(): print(k, v)
import os
HERE = os.path.dirname(os.path.abspath(__file__))
json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in OUT.items()}, open(HERE + "/../verify/nmf_checks.json", "w"), indent=1)
