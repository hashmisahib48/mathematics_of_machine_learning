# Problem 6C.2: tuning alpha on 15 test samples. Writes fig/s6c2.png, verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
from sklearn.model_selection import KFold

alphas = np.logspace(-6, 1, 30)
rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
def predict_all(Xt, yt, Xq):
    """ridge of degree 9 on standardised powers (as in Listing 6.7), all alphas at once; returns (len(alphas), len(Xq))"""
    P = lambda X: X ** np.arange(1, 10); Z = P(Xt); mu, sd = Z.mean(0), Z.std(0); Z = (Z - mu) / sd
    U, s_, Vt = np.linalg.svd(Z, full_matrices=False); b = U.T @ (yt - yt.mean()); Zq = (P(Xq) - mu) / sd
    return np.array([yt.mean() + Zq @ (Vt.T @ (s_ / (s_**2 + a) * b)) for a in alphas])
def data(rng, n): X = rng.uniform(0, 1, (n, 1)); return X, np.sin(2 * np.pi * X[:, 0]) + rng.normal(0, 0.25, n)

# ---- the single experiment of the problem figure (same draws as the book's script)
r = np.random.default_rng(66); Xa, ya = data(r, 60); Xb, yb = data(r, 15); Xc, yc = data(r, 5000)
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
T = [rm(yb, p) for p in predict_all(Xa[:30], ya[:30], Xb)]; F = [rm(yc, p) for p in predict_all(Xa[:30], ya[:30], Xc)]
ref = make_pipeline(PolynomialFeatures(9, include_bias=False), StandardScaler(), Ridge(alpha=alphas[10])).fit(Xa[:30], ya[:30])
print("check against scikit-learn (Ridge pipeline), max |difference| of predictions:", np.abs(ref.predict(Xc) - predict_all(Xa[:30], ya[:30], Xc)[10]).max())
ib = int(np.argmin(T))
print(f"single run: lowest test RMSE {T[ib]:.4f} at alpha = {alphas[ib]:.2e}; fresh RMSE at that alpha {F[ib]:.4f}; best fresh RMSE of all alphas {min(F):.4f}")
print(f"            the 15 test samples understate the error of the chosen model by {F[ib] - T[ib]:.4f} ({100 * (F[ib] - T[ib]) / F[ib]:.1f} per cent)")

# ---- 200 repetitions: tune on the test samples, or tune by cross-validation and touch the test samples once
reps = 200; rows = []
for k in range(reps):
    r = np.random.default_rng(1000 + k); Xt, yt = data(r, 30); Xs, ys = data(r, 15); Xf, yf = data(r, 3000)
    ts = [rm(ys, p) for p in predict_all(Xt, yt, Xs)]; fs = [rm(yf, p) for p in predict_all(Xt, yt, Xf)]
    j = int(np.argmin(ts))                                        # wrong: chosen on the test samples
    cv = np.zeros(len(alphas))
    for tr, va in KFold(5, shuffle=True, random_state=k).split(Xt):
        cv += ((predict_all(Xt[tr], yt[tr], Xt[va]) - yt[va]) ** 2).sum(1)
    i = int(np.argmin(cv))                                        # right: chosen by cross-validation
    rows.append((ts[j], fs[j], ts[i], fs[i]))
R = np.array(rows)
print(f"\n{reps} repetitions (30 training samples, 15 test samples, fresh set of 3000):")
M = R ** 2; se = lambda v: v.std() / np.sqrt(len(v))
for name, c0, c1 in (("chosen on the test samples", 0, 1), ("chosen by cross-validation", 2, 3)):
    d = M[:, c1] - M[:, c0]
    print(f"  {name:27s}: mean squared error reported {M[:, c0].mean():.4f}, true (fresh) {M[:, c1].mean():.4f}; true minus reported {d.mean():+.4f} +- {se(d):.4f}; median RMSE reported {np.median(R[:, c0]):.3f}, true {np.median(R[:, c1]):.3f}; reported below true in {np.mean(R[:, c0] < R[:, c1]):.0%}")
fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.0), dpi=240)
for ax, (cx, cy, ttl, col) in zip(axs, ((1, 0, 'alpha chosen on the 15 test samples', RED), (3, 2, 'alpha chosen by cross-validation', GREEN))):
    ax.plot([0.15, 0.6], [0.15, 0.6], color=GREY, lw=0.7, ls=':'); ax.scatter(R[:, cx], R[:, cy], s=5, color=col, alpha=0.6, lw=0)
    ax.set_xlim(0.15, 0.6); ax.set_ylim(0.1, 0.6); ax.set_xlabel('true RMSE (fresh data)', fontsize=8); ax.set_title(ttl, fontsize=8)
    ax.tick_params(labelsize=7.5); ax.grid(color=GRID, lw=0.5)
axs[0].set_ylabel('RMSE reported on test samples', fontsize=8); fig.tight_layout(w_pad=0.8); save(fig, 'fig/s6c2.png')
