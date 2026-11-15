# Problem 30C.1: the band ignores the readings, the likelihood does not. Writes fig/s30c1.png, prints numbers.
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

x = np.array([1., 2, 3, 4, 5, 6]); X = x[:, None]
data = {'smooth': np.array([0.3, 0.6, 0.9, 1.0, 0.8, 0.5]), 'zigzag': np.array([0.6, -0.6, 0.6, -0.6, 0.6, -0.6])}
g0 = np.linspace(0, 7, 141)[:, None]
fits = {}
for nm, y in data.items():
    g = GPR(C(1.0, 'fixed') * RBF(1.5, 'fixed'), alpha=0.01, optimizer=None).fit(X, y); m, s = g.predict(g0, return_std=True)
    S = g.L_ @ g.L_.T; fit = float(y @ g.alpha_); logdet = float(2 * np.log(np.diag(g.L_)).sum())
    fits[nm] = (m, s)
    i = np.argmin(abs(g0[:, 0] - 3.5))
    print(f"{nm:7s} three terms: -0.5 y^T S^-1 y = {-0.5 * fit:.3f}   -0.5 ln|S| = {-0.5 * logdet:.3f}   -(n/2) ln 2 pi = {-3 * np.log(2 * np.pi):.3f}")
    print(f"{nm:7s} std at 3.5 = {s[i]:.4f}   y^T S^-1 y = {fit:7.2f}   ln|S| = {logdet:.3f}   LML = {g.log_marginal_likelihood_value_:.3f}")
print("std curves identical:", bool(np.allclose(fits['smooth'][1], fits['zigzag'][1])))
for nm, y in data.items():
    ell = np.exp(np.linspace(np.log(0.2), np.log(6), 60)); lm = []
    for l in ell:
        g = GPR(C(1.0, 'fixed') * RBF(l, 'fixed'), alpha=0.01, optimizer=None).fit(X, y); lm.append(g.log_marginal_likelihood_value_)
    lm = np.array(lm); print(f"{nm:7s} best length-scale on the grid (amplitude 1, noise 0.01): {ell[lm.argmax()]:.2f}  LML there {lm.max():.2f};  LML at 1.5 {lm[np.argmin(abs(ell - 1.5))]:.2f}")
    globals()['lm_' + nm] = lm
# standardised residuals: |y_i - LOO mean| / LOO sd
for nm, y in data.items():
    z = []
    for i in range(6):
        k = np.arange(6) != i; g = GPR(C(1.0, 'fixed') * RBF(1.5, 'fixed'), alpha=0.01, optimizer=None).fit(X[k], y[k]); m, s = g.predict(X[i:i + 1], return_std=True)
        z.append((y[i] - m[0]) / np.sqrt(s[0] ** 2 + 0.01))
    print(f"{nm:7s} leave-one-out standardised residuals: {np.round(z, 2).tolist()}   rms {np.sqrt(np.mean(np.square(z))):.2f}")
g = GPR(C(1.0, 'fixed') * RBF(0.2, 'fixed'), alpha=0.01, optimizer=None).fit(X, data['zigzag']); print('std at 3.5 with ell = 0.2:', round(float(g.predict([[3.5]], return_std=True)[1][0]), 4))
print('correlation of neighbours at distance 1, ell = 1.5:', round(float(np.exp(-1 / (2 * 1.5 ** 2))), 4))
fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.9), dpi=240)
ax = axs[0]; ax.plot(g0[:, 0], fits['smooth'][1], color=GREEN, lw=2.6, label='smooth data'); ax.plot(g0[:, 0], fits['zigzag'][1], color=OR, lw=1.3, ls='--', label='zigzag data')
ax.scatter(x, np.zeros(6), s=14, color=BLUE, zorder=5, clip_on=False); ax.set_xlabel('$x_*$'); ax.set_ylabel('posterior std. dev.'); ax.legend(fontsize=8.5, loc='upper center', handlelength=1.6, labelspacing=0.2, borderaxespad=0.1); ax.set_ylim(0, 1.45); ax.tick_params(labelsize=8.5); ax.grid(color=GRID, lw=0.5)
ax = axs[1]; ax.semilogx(ell, lm_smooth, color=GREEN, lw=1.8, label='smooth'); ax.semilogx(ell, lm_zigzag, color=OR, lw=1.8, ls='--', label='zigzag'); ax.axvline(1.5, color=GREY, lw=0.8, ls=':')
ax.set_xlabel('length-scale $\\ell$'); ax.set_ylabel('log marginal likelihood'); ax.set_xticks([0.3, 1, 3]); ax.set_xticklabels(['0.3', '1', '3']); ax.minorticks_off(); ax.legend(fontsize=8.5, loc='lower left', handlelength=1.6, labelspacing=0.2, borderaxespad=0.1); ax.tick_params(labelsize=8.5); ax.grid(color=GRID, lw=0.5)
fig.tight_layout(w_pad=0.8)
print("min font size:", min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()))
save(fig, 'fig/s30c1.png')
