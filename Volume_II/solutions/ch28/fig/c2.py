# Problem 28C.2: does a monotone constraint cost accuracy? Writes fig/s28c2.png, prints verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
os.environ['OMP_NUM_THREADS'] = '1'
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR

grid = np.linspace(0, 10, 500).reshape(-1, 1)
truths = {'rising truth': lambda x: x,
          'truth that falls after 6': lambda x: np.where(x < 6, x, 6 - 1.5 * (x - 6))}
def fit(truth, seed, cst):
    r = np.random.default_rng(seed); x = r.uniform(0, 10, (400, 1)); y = truth(x[:, 0]) + r.normal(0, 2.5, 400)
    return HGBR(monotonic_cst=cst, random_state=0).fit(x, y), x, y

res = {}
for name, f in truths.items():
    e = {None: [], (1,): []}; fall = {None: [], (1,): []}
    for s in range(60):
        for cst in (None, (1,)):
            m, _, _ = fit(f, s, None if cst is None else list(cst)); p = m.predict(grid)
            e[cst].append(np.sqrt(np.mean((p - f(grid[:, 0])) ** 2))); fall[cst].append(int((np.diff(p) < -1e-9).sum()))
    res[name] = e
    a, b = np.array(e[None]), np.array(e[(1,)])
    print(f"{name:26s} RMSE to the truth, mean of 60 data sets: free {a.mean():.3f}  constrained {b.mean():.3f}"
          f"  constrained better in {int((b < a).sum())} of 60;  falling steps (free) {np.mean(fall[None]):.1f}")
print("noise sd 2.5 -> a perfect model has RMSE 2.5 against new noisy y; RMSE above is against the noiseless truth")
# the data set drawn in the book: 48 falling steps for the free model
r = np.random.default_rng(0); xb = r.uniform(0, 10, (400, 1)); yb = xb[:, 0] + r.normal(0, 2.5, 400)
for cst in (None, [1]):
    p = HGBR(monotonic_cst=cst, random_state=0).fit(xb, yb).predict(grid); print("book data, monotonic_cst", cst, " falling steps:", int((np.diff(p) < -1e-9).sum()))

fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.95), dpi=240)
for a_, (name, f) in zip(axs, truths.items()):
    mf, x, y = fit(f, 3, None); mc, _, _ = fit(f, 3, [1])
    a_.scatter(x[:, 0], y, s=2.5, color=GREY, alpha=0.35, zorder=1)
    a_.plot(grid[:, 0], f(grid[:, 0]), color=NAVY, lw=1.1, ls='--', label='truth')
    a_.plot(grid[:, 0], mf.predict(grid), color=OR, lw=1.0, label='free')
    a_.plot(grid[:, 0], mc.predict(grid), color=GREEN, lw=1.4, label='monotone')
    e = res[name]; a_.set_title(f"{name}\nRMSE free {np.mean(e[None]):.2f}, monotone {np.mean(e[(1,)]):.2f}", fontsize=8.3)
    a_.set_xlabel('x'); a_.set_ylabel('y'); a_.grid(color=GRID, lw=0.5); a_.set_axisbelow(True); a_.tick_params(labelsize=8.3)
    a_.xaxis.label.set_size(8.6); a_.yaxis.label.set_size(8.6)
axs[0].legend(fontsize=8.0, loc='upper left', borderaxespad=0.1)
fig.subplots_adjust(wspace=0.32)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s28c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s28c2.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))
