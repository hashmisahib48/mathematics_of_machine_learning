# Problem 16C.2: 100 samples, 40 features, C = inf: separable data.
import sys, os, warnings, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from figstyle import *
from math import comb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
np.set_printoptions(precision=3, suppress=True)
n, p = 100, 40
# Cover (1965): share of the 2^n labellings that a hyperplane with intercept (p + 1 parameters) can produce
share = sum(comb(n - 1, k) for k in range(p + 1)) * 2 / 2 ** n
print("coin-toss labels, n = %d, p = %d: share separable %.4f;  for n = 2(p+1) = %d it is %.2f" %
      (n, p, share, 2 * (p + 1), sum(comb(2 * p + 1, k) for k in range(p + 1)) * 2 / 2 ** (2 * p + 2)))
rng = np.random.default_rng(0)
def make(m, shift=0.5):              # every feature moves by +-shift with the class
    y = rng.integers(0, 2, m); return rng.normal(size=(m, p)) + shift * (2 * y - 1)[:, None], y
X, y = make(n); Xt, yt = make(5000)
def fit(C, tol=1e-4, f=1.0, mi=100000):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always"); m = LogisticRegression(C=C, tol=tol, max_iter=mi).fit(f * X, y)
    return m, [type(r.message).__name__ for r in rec]
print("\nC = inf, standardised-size features: the solver tolerance decides the weights")
print("   tol      n_iter   |w|    train acc  test acc  test log loss  warnings")
for tol in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
    m, w_ = fit(np.inf, tol)
    print(f"{tol:8.0e} {m.n_iter_[0]:7d} {np.linalg.norm(m.coef_):7.2f} {m.score(X, y):8.3f} {m.score(Xt, yt):9.4f} {log_loss(yt, m.predict_proba(Xt)):12.4f}   {w_}")
print("\nthe same data with every feature in units 1000 times smaller (default tol = 1e-4, max_iter = 100)")
m, w_ = fit(np.inf, 1e-4, 0.001, 100); print(f"C = inf: |w| = {np.linalg.norm(m.coef_):.0f}  largest |w_j| = {np.abs(m.coef_).max():.0f}  train acc {m.score(0.001 * X, y):.3f}  n_iter {m.n_iter_[0]}  warnings {w_}")
print("\nwith a penalty (features of the original size)")
for C in (1.0, 0.1, 0.01):
    m, w_ = fit(C); print(f"C = {C:5.2f}: |w| = {np.linalg.norm(m.coef_):5.2f}  train acc {m.score(X, y):.3f}  test acc {m.score(Xt, yt):.4f}  test log loss {log_loss(yt, m.predict_proba(Xt)):.4f}  warnings {w_}")

tols = [10.0 ** -k for k in range(1, 13)]; nrm = {1.0: [], 0.001: []}
for f in nrm:
    for tol in tols: nrm[f].append(np.linalg.norm(fit(np.inf, tol, f)[0].coef_))
print("\nnorm of w for tol = 1e-1 ... 1e-12:", {f: np.round(v, 1).tolist() for f, v in nrm.items()})
fig, ax = plt.subplots(figsize=(3.6, 2.0), dpi=240)
ax.semilogy([-np.log10(t) for t in tols], nrm[1.0], 'o-', color=GREEN, lw=1.5, ms=3, label='original units')
ax.semilogy([-np.log10(t) for t in tols], nrm[0.001], 's--', color=OR, lw=1.4, ms=3, label='units 1000 times smaller')
ax.set_xlabel('$-\\log_{10}$ of the tolerance `tol`'.replace('`', '')); ax.set_ylabel('$\\Vert\\mathbf{w}\\Vert$'); ax.legend(fontsize=8, loc='center right', handlelength=1.8); ax.set_ylim(0.5, 3e5)
fig.tight_layout(); fs = min(t_.get_fontsize() for t_ in fig.findobj(matplotlib.text.Text) if t_.get_text().strip()); save(fig, 'fig/s16c2.png'); print("smallest text", fs)
