# Problem 14C.2: why lbfgs stops at the start on an unscaled feature. Writes fig/s14c2.png, verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
import scipy.optimize as so
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn._loss.loss import HalfPoissonLoss

y = np.array([1., 2, 2, 4, 6]); k = np.arange(5.0)
ll = LinearModelLoss(HalfPoissonLoss(), fit_intercept=True)
def analyse(scale, label):
    x = scale * k; X = x[:, None]; c0 = np.array([0.0, np.log(3.0)])
    f0, g0 = ll.loss_gradient(c0, X, y, None, 0.0, 1)
    H = np.array([[(3 * x ** 2).sum() / 5, (3 * x).sum() / 5], [(3 * x).sum() / 5, 3.0]])
    ev = np.linalg.eigvalsh(H); trials = []
    def fun(c, *a):
        v = ll.loss_gradient(c, *a); trials.append((c[0], v[0])); return v
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)               # the overflow of exp is the point
        r = so.minimize(fun, c0, method='L-BFGS-B', jac=True, options=dict(maxiter=100, maxls=50, gtol=1e-8, ftol=64 * np.finfo(float).eps), args=(X, y, None, 0.0, 1))
    print("== %s (feature = %g x blocks)" % (label, scale))
    print("  start: loss %.4f, gradient (%.4g, %.1e); Hessian eigenvalues %.4g and %.4g, condition number %.3g" % (f0, g0[0], g0[1], ev[0], ev[1], ev[1] / ev[0]))
    print("  Newton step in w: g/H_ww = %.4g" % (-g0[0] / H[0, 0]))
    print("  first L-BFGS-B trial points (w, loss):", [(float(a), float(b)) for a, b in trials[:3]])
    print("  result after %d iteration(s): w = %.4g, loss %.4f; message: %s" % (r.nit, r.x[0], r.fun, r.message))
    return H, trials, r
for sc, lb in ((1e5, "cycles"), (1.0, "10^5 cycles")): analyse(sc, lb)
m = PoissonRegressor(alpha=0, tol=1e-8, max_iter=1000).fit(1e5 * k[:, None], y); print("PoissonRegressor, cycles, lbfgs: n_iter_ = %d, coef_ = %.3g" % (m.n_iter_, m.coef_[0]))
m = PoissonRegressor(alpha=0, tol=1e-8, solver='newton-cholesky').fit(1e5 * k[:, None], y); print("PoissonRegressor, cycles, newton-cholesky: n_iter_ = %d, coef_ x 1e5 = %.4f" % (m.n_iter_, m.coef_[0] * 1e5))
xs = (1e5 * k - (1e5 * k).mean()) / (1e5 * k).std(); m = PoissonRegressor(alpha=0, tol=1e-8, max_iter=1000).fit(xs[:, None], y); print("PoissonRegressor, standardised, lbfgs: n_iter_ = %d, slope per block %.4f" % (m.n_iter_, m.coef_[0] / k.std()))
w_star = PoissonRegressor(alpha=0, tol=1e-12, solver='newton-cholesky').fit(k[:, None], y).coef_[0]; print("true slope per block %.4f" % w_star)

def J(w, x):                                    # loss with the best intercept b = ln(sum y / sum exp(w x))
    b = np.log(y.sum() / np.exp(w * x).sum()); return np.mean(np.exp(b + w * x) - y * (b + w * x))
fig, axs = plt.subplots(1, 2, figsize=(4.6, 1.9), dpi=240)
for ax, sc, ttl, wl in ((axs[0], 1e5, 'feature in cycles', 2.2e-5), (axs[1], 1.0, 'feature in $10^5$ cycles', 2.2)):
    ws = np.linspace(-0.1 * wl, wl, 300); ax.plot(ws / wl, [J(w, sc * k) for w in ws], color=BLUE, lw=1.6)
    ax.plot([0], [J(0, sc * k)], 'o', color=GREY, ms=4.5, zorder=5); ax.plot([w_star / sc / wl], [J(w_star / sc, sc * k)], '*', color=GREEN, ms=9, mec='white', zorder=5)
    ax.set_xlim(-0.1, 1.0)
    if 1 / wl < 1.0: ax.axvline(1 / wl, color=RED, lw=1, ls='--')
    ax.set_title(ttl, fontsize=8.8, pad=3); ax.set_xlabel('weight $w$ (units of the panel)', fontsize=8.8) if False else ax.set_xlabel('$w$', fontsize=9)
    ax.set_ylim(-1.4, 4.2)
axs[0].set_ylabel('loss (best $b$)', fontsize=9)
axs[0].annotate('first trial $w$ = 1,\n230 000 minima\naway', xy=(1.0, 2.2), xytext=(0.12, 2.6), fontsize=8, color=RED, ha='left', va='center', arrowprops=dict(arrowstyle='->', color=RED, lw=0.9)); axs[1].text(0.20, 3.2, 'first trial $w$ = 1', fontsize=8, color=RED, ha='left', va='center')
axs[0].set_xticks([0, 1]); axs[0].set_xticklabels(['0', '2.2e-5']); axs[1].set_xticks([0, 0.5, 1]); axs[1].set_xticklabels(['0', '1.1', '2.2'])
for a in axs: a.grid(color=GRID, lw=0.5); a.tick_params(labelsize=8.2); a.set_axisbelow(True)
fig.tight_layout(w_pad=0.8)
print("distance of the first trial w = 1 from the minimum, in units of the minimum: cycles %.3g, 10^5 cycles %.3g" % (1 / (w_star / 1e5), 1 / w_star))
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s14c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s14c2.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.6 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.6 / w))
