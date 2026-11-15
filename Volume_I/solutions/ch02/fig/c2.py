# Problem 2C.2: digits lost by the normal equations and by QR as cond(X) grows. Writes fig/s2c2.png
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from figstyle import *
n, p = 200, 4
conds = 10.0 ** np.arange(0, 8)
rows = []
for kappa in conds:
    e_ne, e_qr, c_ne = [], [], []
    for rep in range(30):
        r = np.random.default_rng(rep)
        U, _ = np.linalg.qr(r.normal(size=(n, p))); V, _ = np.linalg.qr(r.normal(size=(p, p)))
        X = (U * np.logspace(0, -np.log10(kappa), p)) @ V.T          # singular values from 1 down to 1/kappa
        w = np.ones(p); y = X @ w                                    # exact answer known
        wn = np.linalg.solve(X.T @ X, X.T @ y); Q, R = np.linalg.qr(X); wq = np.linalg.solve(R, Q.T @ y)
        e_ne.append(np.linalg.norm(wn - w) / np.linalg.norm(w)); e_qr.append(np.linalg.norm(wq - w) / np.linalg.norm(w)); c_ne.append(np.linalg.cond(X.T @ X))
    rows.append((kappa, np.median(c_ne), np.median(e_ne), np.median(e_qr)))
print("cond(X)   cond(X^T X)   error normal equations   error QR   digits left: normal, QR")
for k, c2, a, b in rows: print(f"{k:7.0e} {c2:12.1e} {a:20.1e} {b:14.1e}      {-np.log10(a):5.1f} {-np.log10(b):5.1f}")
k4 = [r for r in rows if r[0] == 1e4][0]
print("at cond(X) = 1e4: cond(X^T X) = %.1e; digits lost out of 16: normal equations %.1f, QR %.1f" % (k4[1], 16 + np.log10(k4[2]), 16 + np.log10(k4[3])))
fig, ax = plt.subplots(figsize=(3.6, 2.0), dpi=240)
ax.loglog([r[0] for r in rows], [r[2] for r in rows], '-o', color=RED, ms=3.5, lw=1.4, label='normal equations')
ax.loglog([r[0] for r in rows], [r[3] for r in rows], '--s', color=GREEN, ms=3.5, lw=1.4, label='QR')
ax.loglog(conds, 1.1e-16 * conds ** 2, ':', color=GREY, lw=1, label='$10^{-16}\\kappa^2$'); ax.loglog(conds, 1.1e-16 * conds, '-.', color=GREY, lw=1, label='$10^{-16}\\kappa$')
ax.set_xlabel('cond($\\mathbf{X}$)', fontsize=8.5); ax.set_ylabel('relative error of $\\mathbf{w}$', fontsize=8.5); ax.tick_params(labelsize=7.5); ax.grid(color=GRID, lw=0.5); ax.set_ylim(1e-17, 1)
ax.legend(fontsize=7.5, loc='upper left'); save(fig, 'fig/s2c2.png')
