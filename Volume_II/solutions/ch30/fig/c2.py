# Problem 30C.2: RBF or Matern on a kink. Writes fig/s30c2.png, prints numbers.
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C

def fit(x, y, k): return GPR(C() * k, alpha=0.01, normalize_y=True, random_state=0, n_restarts_optimizer=3).fit(x[:, None], y)
xk = np.linspace(-3, 3, 9); yk = np.abs(xk) + 0.05 * np.random.default_rng(1).normal(size=9)
gg = np.linspace(-4, 4, 801); tr = np.abs(gg); inside = np.abs(gg) <= 3; near = np.abs(gg) <= 0.75
models = {'RBF': RBF(1.0), 'Matern 1/2': Matern(1.0, nu=0.5), 'Matern 3/2': Matern(1.0, nu=1.5), 'Matern 5/2': Matern(1.0, nu=2.5)}
fits = {}
print("nine readings: log marginal likelihood and errors against the true |x| (alpha = 0.01, normalize_y=True)")
for nm, k in models.items():
    g = fit(xk, yk, k); m = g.predict(gg[:, None]); fits[nm] = m
    e = lambda msk: np.sqrt(np.mean((m - tr)[msk] ** 2))
    loo = []
    for i in range(9):
        k = np.arange(9) != i; gi = GPR(g.kernel_, alpha=0.01, normalize_y=True, optimizer=None).fit(xk[k, None], yk[k]); loo.append(gi.predict(xk[i:i + 1, None])[0] - yk[i])
    print(f"{nm:11s} leave-one-out rmse {np.sqrt(np.mean(np.square(loo))):.3f}")
    print(f"{nm:11s} LML {g.log_marginal_likelihood_value_:7.3f}  rmse |x|<=3 {e(inside):.3f}  near kink {e(near):.3f}  |x|>3 {e(~inside):.3f}  value at 0: {g.predict([[0.0]])[0]:.3f}  at 4: {g.predict([[4.0]])[0]:.3f}  kernel_ {g.kernel_}")
print("\nLML(RBF) - LML(Matern 1/2) for n readings on [-3, 3], mean and range over 5 noise seeds")
ns = [9, 17, 33, 65, 129]; D = []
for n in ns:
    d = []
    for s in range(5):
        x = np.linspace(-3, 3, n); y = np.abs(x) + 0.05 * np.random.default_rng(s).normal(size=n)
        d.append(fit(x, y, RBF(1.0)).log_marginal_likelihood_value_ - fit(x, y, Matern(1.0, nu=0.5)).log_marginal_likelihood_value_)
    D.append(d); print(f"n = {n:3d}: {np.mean(d):6.1f}   range {min(d):.1f} to {max(d):.1f}")
D = np.array(D)
fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.9), dpi=240)
ax = axs[0]; ax.plot(gg, tr, color=GREY, lw=0.9, ls=':', label='true $|x|$'); ax.plot(gg, fits['RBF'], color=RED, lw=1.4, ls='--', label='RBF'); ax.plot(gg, fits['Matern 1/2'], color=GREEN, lw=1.4, label='Matérn ½')
ax.scatter(xk, yk, s=14, color=BLUE, edgecolor='white', lw=0.5, zorder=6); ax.set_ylim(-0.6, 5.6); ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.legend(fontsize=8.5, loc='upper center', ncol=1, handlelength=1.6, labelspacing=0.2, borderaxespad=0.1); ax.tick_params(labelsize=8.5); ax.grid(color=GRID, lw=0.5)
ax = axs[1]; ax.semilogx(ns, D.mean(1), 'o-', color=BLUE, ms=3.5, lw=1.4); ax.fill_between(ns, D.min(1), D.max(1), color=LB, lw=0); ax.axhline(0, color=GREY, lw=0.8)
ax.set_xlabel('number of readings $n$'); ax.set_ylabel('LML(RBF) − LML(Matérn ½)'); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns]); ax.minorticks_off(); ax.tick_params(labelsize=8.5); ax.grid(color=GRID, lw=0.5)
fig.tight_layout(w_pad=0.8)
print("min font size:", min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()))
save(fig, 'fig/s30c2.png')
