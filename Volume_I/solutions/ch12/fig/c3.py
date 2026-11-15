# Problem 12C.3: a band that is right on average. Writes fig/s12c3.png
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.linear_model import BayesianRidge, QuantileRegressor

rng = np.random.default_rng(5); nn = 300
x = rng.uniform(0, 10, nn); y = 1 + 0.8 * x + rng.normal(0, 0.2 + 0.35 * x, nn); X = x[:, None]   # the book's data
br = BayesianRidge().fit(X, y); mu, sd = br.predict(X, return_std=True)
inside = np.abs(y - mu) < 2 * sd
print("BayesianRidge: slope %.3f intercept %.3f alpha_ = %.4f -> noise sd %.3f" % (br.coef_[0], br.intercept_, br.alpha_, br.alpha_ ** -0.5))
print("predictive sd over the sample: min %.3f max %.3f ; band half width 2 sd = %.2f everywhere" % (sd.min(), sd.max(), 2 * sd.mean()))
print("inside +-2 sd: all %.3f, x < 2.5 %.3f (n = %d), x > 7.5 %.3f (n = %d)" % (inside.mean(), inside[x < 2.5].mean(), (x < 2.5).sum(), inside[x > 7.5].mean(), (x > 7.5).sum()))
q = {p: QuantileRegressor(quantile=p, alpha=0, solver='highs').fit(X, y) for p in (0.025, 0.975)}
lo, hi = q[0.025].predict(X), q[0.975].predict(X); ins = (y > lo) & (y < hi)
print("QuantileRegressor 0.025 / 0.975: inside all %.3f, x < 2.5 %.3f, x > 7.5 %.3f" % (ins.mean(), ins[x < 2.5].mean(), ins[x > 7.5].mean()))
edges = [0, 2.5, 5, 7.5, 10]
print("x range      n   true sd(mid)  residual sd   BR half-width  BR inside  QR half-width  QR inside")
rows = []
for a, b in zip(edges[:-1], edges[1:]):
    m = (x >= a) & (x < b); mid = (a + b) / 2
    row = (a, b, m.sum(), 0.2 + 0.35 * mid, np.std((y - mu)[m]), 2 * sd[m].mean(), inside[m].mean(), (hi - lo)[m].mean() / 2, ins[m].mean()); rows.append(row)
    print("%4.1f - %4.1f %4d %11.2f %12.2f %14.2f %10.2f %14.2f %10.2f" % row)
print("true sd at x = 0 and x = 10: %.2f and %.2f, ratio %.1f" % (0.2, 3.7, 3.7 / 0.2))

fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.0), dpi=240, gridspec_kw={'width_ratios': [1.3, 1]})
ax = axs[0]; xs = np.linspace(0, 10, 100); m_, s_ = br.predict(xs[:, None], return_std=True)
ax.scatter(x, y, s=3, color=BLUE, lw=0, alpha=0.6); ax.fill_between(xs, m_ - 2 * s_, m_ + 2 * s_, color='#CFE8DA', lw=0, alpha=0.8); ax.plot(xs, m_, color=GREEN, lw=1.3)
ax.plot(xs, q[0.025].predict(xs[:, None]), color=RED, lw=1.2, ls='--'); ax.plot(xs, q[0.975].predict(xs[:, None]), color=RED, lw=1.2, ls='--')
ax.set_xlabel('$x$', fontsize=9); ax.set_ylabel('$y$', fontsize=9); ax.tick_params(labelsize=8.5); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax = axs[1]; mids = [(r[0] + r[1]) / 2 for r in rows]
ax.plot(mids, [r[6] for r in rows], '-o', color=GREEN, ms=4, lw=1.4, label='Bayesian band'); ax.plot(mids, [r[8] for r in rows], '--s', color=RED, ms=4, lw=1.4, label='quantile lines')
ax.axhline(0.95, color=GREY, lw=0.8, ls=':'); ax.set_ylim(0.75, 1.03); ax.set_xlabel('$x$ (centre of the bin)', fontsize=8.5); ax.set_ylabel('share inside', fontsize=8.5); ax.tick_params(labelsize=8.5)
ax.legend(fontsize=8.5, loc='lower left', borderpad=0.2, handlelength=1.6, labelspacing=0.2); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(pad=0.4, w_pad=0.8)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s12c3.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s12c3.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))
