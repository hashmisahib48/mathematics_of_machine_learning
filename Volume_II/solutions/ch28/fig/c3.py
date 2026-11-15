# Problem 28C.3: "a hundred times faster". Writes fig/s28c3.png, prints verify/c3.out
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
os.environ['OMP_NUM_THREADS'] = '1'
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier as GBC, HistGradientBoostingClassifier as HGBC
from threadpoolctl import threadpool_limits

sp = json.load(open('../../ch28/verify/speed.json'))            # the book's timings (one thread)
n = np.array(sp['n']); gb = np.array([np.nan if v is None else v for v in sp['gbc']]); hg = np.array(sp['hgbc'])
ok = ~np.isnan(gb); ratio = gb[ok] / hg[ok]
print("book:  n", n[ok].tolist(), " ratio", ratio.round(1).tolist())

def clock(est, X, y, reps=2):
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); est.fit(X, y); t.append(time.perf_counter() - t0)
    return min(t)
print("\nrerun on this computer, one thread, best of 2 (defaults of both, 100 trees):")
print("      n   GBC s   HGBC s (no early stop)  ratio   HGBC defaults: s  trees   test acc GBC / HGBC")
mine = {}
with threadpool_limits(limits=1):
    for nn in (1000, 3000, 10000):
        X, y = make_classification(2 * nn, 20, n_informative=8, random_state=0); Xa, ya, Xb, yb = X[:nn], y[:nn], X[nn:], y[nn:]
        g = GBC(random_state=0); tg = clock(g, Xa, ya)
        h = HGBC(early_stopping=False, random_state=0); th = clock(h, Xa, ya)
        hd = HGBC(random_state=0); td = clock(hd, Xa, ya)
        mine[nn] = (tg, th); print(f"{nn:7d} {tg:7.2f} {th:8.2f} {tg / th:21.1f} {td:20.2f} {hd.n_iter_:6d} {g.score(Xb, yb):10.4f} / {h.score(Xb, yb):.4f}")
    for nn in (30000, 100000):
        X, y = make_classification(nn, 20, n_informative=8, random_state=0)
        hd = HGBC(random_state=0); td = clock(hd, X, y, 1)
        h = HGBC(early_stopping=False, random_state=0); th = clock(h, X, y, 1)
        print(f"{nn:7d}   HGBC with defaults: {td:.2f} s, {hd.n_iter_} trees (early stopping {hd.do_early_stopping_});  without: {th:.2f} s, {h.n_iter_} trees")
# how the ratio might grow: straight line through the log-log points of the last three sizes
lx = np.log(n[ok][-3:]); sl_g = np.polyfit(lx, np.log(gb[ok][-3:]), 1); sl_h = np.polyfit(lx, np.log(hg[ok][-3:]), 1)
print(f"\nlog-log slope of GBC time {sl_g[0]:.2f}, of HGBC time {sl_h[0]:.2f}  -> ratio grows like n^{sl_g[0] - sl_h[0]:.2f}")
nn = np.exp(np.log(30000) + (np.log(100) - np.log(ratio[-1])) / (sl_g[0] - sl_h[0]))
print(f"ratio 100 would be reached at about n = {nn:,.0f} rows;  predicted ratio at 100000 rows: {np.exp(np.polyval(sl_g, np.log(1e5)) - np.polyval(sl_h, np.log(1e5))):.0f}")
print(f"predicted GBC time at 100000 rows: {np.exp(np.polyval(sl_g, np.log(1e5))):.0f} s;  HGBC measured {hg[4]} s")

fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.9), dpi=240)
a = axs[0]; a.plot(n[ok], gb[ok], '-o', color=OR, ms=3, lw=1.1, label='GradientBoosting'); a.plot(n, hg, '-s', color=GREEN, ms=3, lw=1.1, label='HistGradientBoosting')
a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('rows n'); a.set_ylabel('seconds'); a.legend(fontsize=8.0, loc='upper left', borderaxespad=0.1)
b = axs[1]; xs = np.geomspace(1000, 4e5, 50); b.plot(n[ok], ratio, '-o', color=PUR, ms=3, lw=1.1, label='measured')
b.plot(xs, np.exp(np.polyval(sl_g, np.log(xs)) - np.polyval(sl_h, np.log(xs))), ls='--', color=GREY, lw=0.9, label='straight-line guess')
b.axhline(100, color=RED, lw=0.8); b.text(1100, 112, 'claim: 100', fontsize=8.0, color=RED)
b.set_xscale('log'); b.set_yscale('log'); b.set_ylim(1, 400); b.set_xlabel('rows n'); b.set_ylabel('time ratio'); b.legend(fontsize=8.0, loc='lower right', borderaxespad=0.1)
for a in axs: a.grid(color=GRID, lw=0.5); a.set_axisbelow(True); a.tick_params(labelsize=8.3); a.xaxis.label.set_size(8.6); a.yaxis.label.set_size(8.6)
fig.subplots_adjust(wspace=0.38)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s28c3.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s28c3.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))
