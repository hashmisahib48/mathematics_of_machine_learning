# Problem 36C.1: why two centres cannot separate two rings. Writes fig/s36c1.png, verify/c1.out
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.datasets import make_circles
from sklearn.metrics import adjusted_rand_score

X, y = make_circles(300, factor=0.4, noise=0.05, random_state=0)      # the book's data
km = KMeans(2, n_init=10, random_state=0).fit(X)
rad = np.linalg.norm(X, axis=1)
print("radius of the points: inner ring mean %.3f (min %.3f max %.3f), outer ring mean %.3f (min %.3f max %.3f)"
      % (rad[y == 1].mean(), rad[y == 1].min(), rad[y == 1].max(), rad[y == 0].mean(), rad[y == 0].min(), rad[y == 0].max()))
print("2-means: centres", km.cluster_centers_.round(3).tolist(), " inertia %.2f  ARI %.3f" % (km.inertia_, adjusted_rand_score(y, km.labels_)))
print("share of each ring in cluster 0: outer %.2f  inner %.2f" % ((km.labels_[y == 0] == 0).mean(), (km.labels_[y == 1] == 0).mean()))
# the correct partition, with its own means
J_true = sum(((X[y == g] - X[y == g].mean(0)) ** 2).sum() for g in (0, 1))
print("inertia of the true partition (rings): %.2f   inertia of the 2-means solution: %.2f" % (J_true, km.inertia_))
print("means of the two rings:", X[y == 0].mean(0).round(3).tolist(), X[y == 1].mean(0).round(3).tolist())
# best straight line of all: scan direction and offset
best = (0, None)
for th in np.linspace(0, np.pi, 361):
    u = np.array([np.cos(th), np.sin(th)]); s = X @ u
    for b in np.linspace(-1.1, 1.1, 221):
        lab = (s > b).astype(int); acc = max((lab == y).mean(), (lab != y).mean())
        if acc > best[0]: best = (acc, (th, b, lab))
th, b, labL = best[1]
print("best accuracy of any straight line (angle %.1f deg, offset %.2f): %.3f   ARI %.3f   (all in one cluster: 0.500)"
      % (np.degrees(th), b, best[0], adjusted_rand_score(y, labL)))
with warnings.catch_warnings():             # the neighbour graph has two components: that is the point
    warnings.simplefilter('ignore', UserWarning)
    sp = SpectralClustering(2, affinity='nearest_neighbors', n_neighbors=15, random_state=0).fit(X)
db = DBSCAN(eps=0.15, min_samples=5).fit(X)
print("SpectralClustering(2, nearest_neighbors, 15): ARI %.3f   DBSCAN(eps=0.15): ARI %.3f  clusters %d"
      % (adjusted_rand_score(y, sp.labels_), adjusted_rand_score(y, db.labels_), len(set(db.labels_) - {-1})))

fig, axs = plt.subplots(1, 3, figsize=(5.4, 1.95), dpi=240)
def draw(ax, lab, title, line=None, cen=None):
    ax.scatter(X[:, 0], X[:, 1], c=np.where(lab == 0, BLUE, OR), s=4, lw=0)
    if line is not None:
        u = np.array([np.cos(line[0]), np.sin(line[0])]); p0 = u * line[1]; d = np.array([-u[1], u[0]])
        ax.plot(*np.array([p0 - 1.6 * d, p0 + 1.6 * d]).T, color=NAVY, lw=0.9)
    if cen is not None: ax.scatter(*cen.T, marker='X', s=36, color=GREEN, edgecolor='white', lw=0.6, zorder=5)
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8.6, pad=3)
    for s in ax.spines.values(): s.set_visible(False)
c = km.cluster_centers_; v = c[1] - c[0]; thk = np.arctan2(v[1], v[0]); bk = ((c[0] + c[1]) / 2) @ (v / np.linalg.norm(v))
draw(axs[0], km.labels_, 'K-means, K = 2\nARI %.3f' % adjusted_rand_score(y, km.labels_), (thk, bk), c)
draw(axs[1], labL, 'best straight line of all\nARI %.3f' % adjusted_rand_score(y, labL), (th, b))
draw(axs[2], sp.labels_, 'spectral clustering\nARI %.3f' % adjusted_rand_score(y, sp.labels_))
fig.subplots_adjust(wspace=0.05, left=0.01, right=0.99, top=0.82, bottom=0.01)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s36c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s36c1.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))
