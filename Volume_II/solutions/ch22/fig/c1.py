# Problem 22C.1: moving a seed. Writes fig/s22c1.png, prints to verify/c1.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + "/../../../lib"); os.chdir(HERE + "/..")
import warnings; warnings.simplefilter("error", FutureWarning); warnings.simplefilter("error", DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image

from sklearn.svm import SVC
X8 = np.array([[1, 5], [2, 3], [3, 4], [4, 1], [7, 6], [8, 2], [4, 7], [6, 3]], float); y8 = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
fit = lambda X: SVC(kernel="linear", C=1e6, tol=1e-10).fit(X, y8)
m0 = fit(X8); w0, b0 = m0.coef_[0], float(m0.intercept_[0])
print("old street: w", w0.round(4), "b", round(b0, 4), "width", round(2 / np.linalg.norm(w0), 4), "support seeds", m0.support_ + 1)
print("old scores f of seeds 1-8:", (X8 @ w0 + b0).round(2))
cases = [(5, [9, 5], "seed 6 to (9, 5)"), (3, [5, 2], "seed 4 to (5, 2)")]
res = []
for idx, new, nm in cases:
    X = X8.copy(); X[idx] = new; m = fit(X); w, b = m.coef_[0], float(m.intercept_[0])
    yf_old = y8[idx] * (w0 @ np.array(new) + b0)
    print("%s: y f of the OLD line at the new place = %.2f -> %s" % (nm, yf_old, "street must change" if yf_old < 1 - 1e-9 else "old solution still optimal"))
    print("   new w", w.round(4), "b", round(b, 4), "width %.4f" % (2 / np.linalg.norm(w)), "support seeds", m.support_ + 1)
    res.append((X, w, b, m))
# how far may seed 4 move along the line towards (5, 2) before the street changes? scan
d = np.array([1., 1.])
for t in (0.0, 0.2, 0.3, 1 / 3, 0.35, 0.5, 1.0):
    X = X8.copy(); X[3] = X8[3] + t * d; m = fit(X)
    print("seed 4 moved by %.2f x (1, 1): y f old line %.2f, w %s, width %.4f" % (t, -(w0 @ X[3] + b0), m.coef_[0].round(4), 2 / np.linalg.norm(m.coef_)))
fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.0), dpi=240)
for ax, (idx, new, nm), (X, w, b, m) in zip(axs, cases, res):
    xx = np.array([0, 10.0])
    for c, ls, lw in ((0, "-", 1.2), (1, "--", 0.7), (-1, "--", 0.7)): ax.plot(xx, (c - b0 - w0[0] * xx) / w0[1], color=GRID if False else GREY, ls=ls, lw=lw, alpha=0.8)
    for c, ls, lw in ((0, "-", 1.4), (1, "--", 0.8), (-1, "--", 0.8)): ax.plot(xx, (c - b - w[0] * xx) / w[1], color=GREEN, ls=ls, lw=lw)
    for cls, col, mk in ((-1, BLUE, "o"), (1, OR, "s")):
        mm = y8 == cls; ax.scatter(X[mm, 0], X[mm, 1], s=14, color=col, marker=mk, edgecolor="white", lw=0.5, zorder=5)
    ax.scatter(*X[idx], s=40, facecolor="none", edgecolor=RED, lw=1.0, zorder=6)
    arr(ax, tuple(X8[idx]), tuple(new), c=RED, lw=1.0, ms=6)
    ax.set_xlim(0, 10); ax.set_ylim(0, 8.5); ax.set_aspect("equal"); ax.tick_params(labelsize=8); ax.set_title(nm + ", width %.2f" % (2 / np.linalg.norm(w)), fontsize=8.6, pad=3)
fig.subplots_adjust(wspace=0.12, left=0.04, right=0.99, top=0.9, bottom=0.08)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig("fig/s22c1.png", dpi=240, bbox_inches="tight", pad_inches=0.04, facecolor="white")
with Image.open("fig/s22c1.png") as im: w_ = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w_, fs, fs * 4.9 / w_))
