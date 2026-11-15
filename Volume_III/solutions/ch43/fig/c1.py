# Problem 43C.1: the exact factorisations of the three paints. Writes fig/s43c1.png, verify/c1.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from matplotlib.patches import Wedge
from PIL import Image

X = np.array([[4, 2], [3, 3], [2, 4.]])
lo, hi = np.degrees(np.arctan2(1, 2)), np.degrees(np.arctan2(2, 1))          # 26.57 and 63.43 degrees
print("tin angles:", [round(float(np.degrees(np.arctan2(r[1], r[0]))), 2) for r in X], " limits", round(float(lo), 2), round(float(hi), 2))
def minW(t1, t2):                                                             # smallest entry of W = X H^-1
    H = np.array([[np.cos(np.radians(t1)), np.sin(np.radians(t1))], [np.cos(np.radians(t2)), np.sin(np.radians(t2))]])
    return (X @ np.linalg.inv(H)).min()
ths = np.linspace(0, 45, 4501); mw = np.array([minW(t, 90) for t in ths])
print("second ray at 90 deg: smallest entry of W is >= 0 for angles up to %.2f deg" % ths[np.where(mw >= -1e-12)[0].max()])
print("smallest entry at 0 deg: %.3f, at 26.57 deg: %.3f, at 40 deg: %.3f" % (minW(0, 90), minW(lo, 90), minW(40, 90)))
A1 = np.arange(0, 45, 0.5); B1 = np.arange(45.5, 90.01, 0.5)
ok = np.array([[minW(a, b) >= -1e-12 for b in B1] for a in A1]); pred = (A1[:, None] <= lo) & (B1[None, :] >= hi)
print("grid of %d pairs of rays: %d give a non-negative W; the rule 'first ray <= 26.57, second >= 63.43' predicts %d, and the two sets agree: %s"
      % (ok.size, ok.sum(), pred.sum(), bool((ok == pred).all())))

fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.1), dpi=240, gridspec_kw={'width_ratios': [1.1, 1]})
ax = axs[0]
ax.add_patch(Wedge((0, 0), 5.2, 0, lo, color='#FCE8C8', lw=0)); ax.add_patch(Wedge((0, 0), 5.2, hi, 90, color='#FCE8C8', lw=0))
ax.add_patch(Wedge((0, 0), 5.2, lo, hi, color='#E3F4EA', lw=0))
ax.scatter(X[:, 0], X[:, 1], s=22, color=BLUE, edgecolor='white', lw=0.6, zorder=5)
for h in ([2, 1], [1, 2]): arr(ax, (0, 0), (h[0] * 1.9, h[1] * 1.9), c=GREEN, lw=1.5, ms=7)
arr(ax, (0, 0), (4.9, 0), c=OR, lw=1.1, ms=7); arr(ax, (0, 0), (0, 4.9), c=OR, lw=1.1, ms=7)
ax.text(5.0, 0.45, 'first ray', fontsize=7.8, color=NAVY, ha='right'); ax.text(0.1, 4.75, 'second ray', fontsize=7.8, color=NAVY, ha='left', va='center')
ax.text(2.4, 3.15, 'tins', fontsize=7.8, color=BLUE); ax.set_xlim(-0.2, 5.4); ax.set_ylim(-0.2, 5.4); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)
ax = axs[1]
ax.plot(ths, mw, color=PUR, lw=1.4); ax.axhline(0, color=GREY, lw=0.8); ax.axvline(lo, color=GREEN, lw=1.0, ls='--')
ax.text(lo + 0.8, ax.get_ylim()[0] * 0 + 1.1, '26.57°', fontsize=7.8, color=GREEN)
ax.set_xlabel('angle of the first ray (degrees; second ray fixed at 90°)', fontsize=8); ax.set_ylabel('smallest entry of W', fontsize=8)
ax.tick_params(labelsize=7.8); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.subplots_adjust(wspace=0.28, left=0.02, right=0.99, top=0.97, bottom=0.2)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s43c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s43c1.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))
