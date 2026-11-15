# Problem 14C.1: exp of a fit to ln y against the gamma GLM. Writes fig/s14c1.png, verify/c1.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from scipy.special import digamma
from sklearn.linear_model import GammaRegressor

def data(seed, n=70):                                           # the book's generator (seed 8)
    rng = np.random.default_rng(seed); xs = rng.uniform(0, 4, n); return xs, rng.gamma(1.2, np.exp(0.6 + 0.55 * xs) / 1.2)
def fits(xs, ys):
    g = GammaRegressor(alpha=0, tol=1e-10, max_iter=500).fit(xs[:, None], ys); c = np.polyfit(xs, np.log(ys), 1); return g, c
xs, ys = data(8); g, c = fits(xs, ys)
print("book data set at x = 4: gamma GLM %.2f, exp(fit to ln y) %.2f, true mean %.2f" % (g.predict([[4.]])[0], np.exp(c[1] + 4 * c[0]), np.exp(0.6 + 0.55 * 4)))
print("coefficients: gamma GLM intercept %.3f slope %.3f; log-OLS intercept %.3f slope %.3f; truth 0.600 and 0.550" % (g.intercept_, g.coef_[0], c[1], c[0]))
k = 1.2; print("E[ln y] = ln mu + psi(k) - ln k, with psi(1.2) = %.4f and ln 1.2 = %.4f: shift %.4f, factor exp(shift) = %.4f" % (digamma(k), np.log(k), digamma(k) - np.log(k), np.exp(digamma(k) - np.log(k))))
print("total of the observations %.1f; predicted totals: gamma GLM %.1f, exp(log-OLS) %.1f" % (ys.sum(), g.predict(xs[:, None]).sum(), np.exp(c[1] + c[0] * xs).sum()))
print("smallest observations:", np.round(np.sort(ys)[:4], 3), " ln of them:", np.round(np.log(np.sort(ys)[:4]), 2), " ln of the largest %.2f" % np.log(ys.max()))
rg, rl = [], []
for s in range(300):
    a, b = data(2000 + s); gg, cc = fits(a, b); t = np.exp(0.6 + 0.55 * 2.0); rg.append(gg.predict([[2.0]])[0] / t); rl.append(np.exp(cc[1] + 2.0 * cc[0]) / t)
rg, rl = np.array(rg), np.array(rl)
print("300 data sets, prediction / true mean at x = 2: gamma GLM median %.3f (10th-90th %.3f-%.3f); exp(log-OLS) median %.3f (%.3f-%.3f)" % (np.median(rg), *np.percentile(rg, [10, 90]), np.median(rl), *np.percentile(rl, [10, 90])))
print("share of data sets in which the log-OLS prediction is below the true mean: %.3f; gamma GLM: %.3f" % (np.mean(rl < 1), np.mean(rg < 1)))

fig, axs = plt.subplots(1, 2, figsize=(4.6, 1.9), dpi=240)
ax = axs[0]; xx = np.linspace(0, 4, 100); ax.scatter(xs, ys, s=7, color=BLUE, alpha=0.8, zorder=3)
ax.plot(xx, np.exp(0.6 + 0.55 * xx), color=NAVY, lw=1.1, ls=':', label='true mean'); ax.plot(xx, g.predict(xx[:, None]), color=GREEN, lw=1.6, label='gamma GLM')
ax.plot(xx, np.exp(c[1] + c[0] * xx), color=OR, lw=1.6, ls='--', label='exp of fit to ln $y$'); ax.set_ylim(0, 32); ax.set_xlabel('$x$', fontsize=9); ax.set_ylabel('$y$', fontsize=9)
ax.legend(fontsize=8, loc='upper left', borderpad=0.1, labelspacing=0.25, handlelength=1.8)
ax = axs[1]; bins = np.linspace(0.2, 1.6, 29); ax.hist(rl, bins=bins, color=OR, alpha=0.9, label='exp of fit to ln $y$'); ax.hist(rg, bins=bins, color=GREEN, alpha=0.8, label='gamma GLM')
ax.axvline(1, color=NAVY, lw=1, ls='--'); ax.set_xlabel('prediction / true mean at $x$ = 2', fontsize=9); ax.set_ylabel('data sets (of 300)', fontsize=9); ax.set_ylim(0, 1.5 * ax.get_ylim()[1]); ax.legend(fontsize=8, loc='center right', borderpad=0.1, labelspacing=0.25, handlelength=1.2)
for a in axs: a.grid(color=GRID, lw=0.5); a.tick_params(labelsize=8.2); a.set_axisbelow(True)
fig.tight_layout(w_pad=0.8)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s14c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s14c1.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.6 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.6 / w))
