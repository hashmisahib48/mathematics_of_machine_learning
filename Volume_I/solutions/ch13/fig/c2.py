# Problem 13C.2: a band from 30 points. Writes fig/s13c2.png, verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.linear_model import QuantileRegressor

rng = np.random.default_rng(11)                                # the book's generator, same order of draws
def mk(n): x_ = rng.uniform(0, 10, n); return x_[:, None], 2 + 1.5 * x_ + rng.normal(size=n) * (0.3 + 0.35 * x_)
def lines(Xs, ys, taus): return [QuantileRegressor(quantile=t, alpha=0).fit(Xs, ys) for t in taus]
Xs, ys = mk(30); Xt, yt = mk(4000)
lo, hi = lines(Xs, ys, (0.05, 0.95))
print("the book's data set (n = 30): below the 0.05 line %.3f, below the 0.95 line %.3f, band %.3f" % (np.mean(yt <= lo.predict(Xt)), np.mean(yt <= hi.predict(Xt)), np.mean(yt <= hi.predict(Xt)) - np.mean(yt <= lo.predict(Xt))))
for nm, q in (("0.05", lo), ("0.95", hi)):
    p = q.predict(Xs); print("  line %s: on the line %d points, strictly below %d, above %d; in-sample share at or below %.3f" % (nm, (abs(ys - p) < 1e-9).sum(), (ys < p - 1e-9).sum(), (ys > p + 1e-9).sum(), np.mean(ys <= p + 1e-9)))
print("  expected number of the 30 targets below the 0.05 line: %.1f; the line is held by the %d lowest" % (0.05 * 30, 2))
# true conditional quantiles for comparison: y = 2 + 1.5 x + z (0.3 + 0.35 x), z ~ N(0, 1)
xg = np.array([0., 5., 10.]); zq = 1.6449
print("true 0.05 / 0.95 quantile at x = 0, 5, 10:", np.round(2 + 1.5 * xg - zq * (0.3 + 0.35 * xg), 2), np.round(2 + 1.5 * xg + zq * (0.3 + 0.35 * xg), 2))
print("fitted 0.05 / 0.95 line at x = 0, 5, 10:   ", np.round(lo.predict(xg[:, None]), 2), np.round(hi.predict(xg[:, None]), 2))

R = 100; ns = (30, 60, 120, 300, 1000); cov = {}; cross = {}; tails = {}
for n in ns:
    c = []; cr = 0; tl = []
    for _ in range(R):
        A, b = mk(n); T, t = mk(4000); l, h = lines(A, b, (0.05, 0.95))
        pl, ph = l.predict(T), h.predict(T); c.append(np.mean(t <= ph) - np.mean(t <= pl)); tl.append((np.mean(t <= pl), np.mean(t <= ph)))
        g = np.linspace(0, 10, 101)[:, None]; cr += int(np.any(l.predict(g) > h.predict(g)))
    cov[n] = np.array(c); cross[n] = cr / R; tails[n] = np.array(tl).mean(0)
    print("n = %4d: band coverage median %.3f, 10th-90th percentile %.3f - %.3f, share of sets below 0.85: %.2f, mean tails %.3f / %.3f, lines crossed on [0,10]: %.2f"
          % (n, np.median(c), *np.percentile(c, [10, 90]), np.mean(np.array(c) < 0.85), *tails[n], cross[n]))
# remedy: ask for 0.025 and 0.975 when the data set is small
c = []
for _ in range(R):
    A, b = mk(30); T, t = mk(4000); l, h = lines(A, b, (0.025, 0.975)); c.append(np.mean(t <= h.predict(T)) - np.mean(t <= l.predict(T)))
c = np.array(c); print("n = 30 with levels 0.025 / 0.975: band coverage median %.3f, 10th-90th percentile %.3f - %.3f, mean %.3f" % (np.median(c), *np.percentile(c, [10, 90]), c.mean()))

fig, axs = plt.subplots(1, 2, figsize=(4.6, 1.9), dpi=240)
ax = axs[0]; bins = np.linspace(0.6, 1.0, 21)
ax.hist(cov[30], bins=bins, color=OR, alpha=0.85, label='n = 30'); ax.hist(cov[300], bins=bins, color=BLUE, alpha=0.85, label='n = 300')
ax.axvline(0.9, color=NAVY, lw=1, ls='--'); ax.set_xlabel('coverage of the nominal 90 % band', fontsize=8.6); ax.set_ylabel('data sets (of %d)' % R, fontsize=8.6); ax.legend(fontsize=8.2, loc='upper left', borderpad=0.1, handlelength=1.2)
ax = axs[1]; med = [np.median(cov[n]) for n in ns]; p10 = [np.percentile(cov[n], 10) for n in ns]; p90 = [np.percentile(cov[n], 90) for n in ns]
ax.fill_between(ns, p10, p90, color=LB); ax.plot(ns, med, '-o', color=BLUE, ms=3.5, lw=1.5); ax.axhline(0.9, color=NAVY, lw=1, ls='--')
ax.set_xscale('log'); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns]); ax.minorticks_off(); ax.set_xlabel('training points n', fontsize=8.6); ax.set_ylabel('coverage', fontsize=8.6)
ax.text(33, 0.755, 'median and\n10 to 90 %', fontsize=8.2, color=BLUE, va='bottom')
for a in axs: a.grid(color=GRID, lw=0.5); a.tick_params(labelsize=8.2); a.set_axisbelow(True)
fig.tight_layout(w_pad=0.8)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s13c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s13c2.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.6 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.6 / w))
