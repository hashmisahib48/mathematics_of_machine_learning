# Problem 22C.2: gamma = 100 gives isolated islands. Writes fig/s22c2.png, prints to verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + "/../../../lib"); os.chdir(HERE + "/..")
import warnings; warnings.simplefilter("error", FutureWarning); warnings.simplefilter("error", DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image

from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
X, y = make_moons(n_samples=120, noise=0.3, random_state=1)
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)   # the book's split
m = SVC(C=100, gamma=100).fit(Xa, ya)
print("C=100 gamma=100: train %.3f test %.3f support vectors %d of %d, intercept b = %.3f" % (m.score(Xa, ya), m.score(Xb, yb), len(m.support_), len(ya), m.intercept_[0]))
D = euclidean_distances(Xa); np.fill_diagonal(D, np.inf)
print("distance to the nearest other training sample: median %.3f, mean %.3f; range of one bump 1/sqrt(gamma) = %.3f" % (np.median(D.min(1)), D.min(1).mean(), 100 ** -0.5))
print("kernel value between a sample and its nearest neighbour: median %.4f" % np.median(np.exp(-100 * D.min(1) ** 2)))
Kt = rbf_kernel(Xb, Xa, gamma=100); kmax = Kt.max(1)
print("test samples: largest kernel value to any training sample: median %.3f; share below 0.1: %.2f, below 0.5: %.2f" % (np.median(kmax), (kmax < 0.1).mean(), (kmax < 0.5).mean()))
f = m.decision_function(Xb); far = kmax < 0.1
print("those far test samples: predicted classes", np.unique(m.predict(Xb[far]), return_counts=True), " |f - b| max %.3f" % np.abs(f[far] - m.intercept_[0]).max())
print("test accuracy on near samples (kmax >= 0.1): %.3f   on far samples: %.3f" % ((m.predict(Xb[~far]) == yb[~far]).mean(), (m.predict(Xb[far]) == yb[far]).mean()))
print("alternatives (C, gamma): test accuracy, support vectors")
for C, g in ((100, 100), (100, 1), (1, 1), (1, "scale"), (10, 0.5), (1, 10)):
    mm = SVC(C=C, gamma=g).fit(Xa, ya); print("  C = %-4g gamma = %-6s train %.3f  test %.3f  SV %d" % (C, g, mm.score(Xa, ya), mm.score(Xb, yb), len(mm.support_)))
print("gamma = 'scale' is", round(1 / (2 * Xa.var()), 4))
gx, gy = np.meshgrid(np.linspace(-1.6, 2.6, 220), np.linspace(-1.2, 1.7, 220)); G = np.c_[gx.ravel(), gy.ravel()]
fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.95), dpi=240)
for ax, g, ttl in ((axs[0], 100, "gamma = 100:  range 0.1"), (axs[1], 1, "gamma = 1:  range 1")):
    km = rbf_kernel(G, Xa, gamma=g).max(1).reshape(gx.shape)
    im = ax.contourf(gx, gy, km, levels=[0, 0.1, 0.37, 0.6, 1.01], colors=["#FFFFFF", "#DCE8F8", "#A9C4EA", "#6F9BD8"])
    ax.contour(gx, gy, km, levels=[np.exp(-1)], colors=[NAVY], linewidths=0.6)
    for cls, col, mk in ((0, BLUE, "o"), (1, OR, "s")):
        mm = ya == cls; ax.scatter(Xa[mm, 0], Xa[mm, 1], s=7, color=col, marker=mk, edgecolor="white", lw=0.3, zorder=5)
    ax.set_xlim(-1.6, 2.6); ax.set_ylim(-1.2, 1.7); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl, fontsize=8.6, pad=3)
    for s in ax.spines.values(): s.set_visible(False)
fig.subplots_adjust(wspace=0.05, left=0.01, right=0.99, top=0.88, bottom=0.01)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig("fig/s22c2.png", dpi=240, bbox_inches="tight", pad_inches=0.04, facecolor="white")
with Image.open("fig/s22c2.png") as im: w_ = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w_, fs, fs * 4.9 / w_))
