# Problem 44C.2: a flat ring (annulus), the methods of the chapter. Writes fig/s44c2.png, verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from scipy.spatial import procrustes
from sklearn.manifold import ClassicalMDS, Isomap, LocallyLinearEmbedding, SpectralEmbedding, TSNE, trustworthiness
from sklearn.neighbors import NearestNeighbors

rng = np.random.RandomState(1); n = 600
r = np.sqrt(rng.uniform(1, 4, n)); a = rng.uniform(0, 2 * np.pi, n); X = np.c_[r * np.cos(a), r * np.sin(a)]   # radii 1 to 2, hole of radius 1
print("annulus: %d points, radii %.2f to %.2f" % (n, r.min(), r.max()))
nn10 = NearestNeighbors(n_neighbors=11).fit(X).kneighbors(X)[0]; print("mean distance to the 10th neighbour %.3f (hole diameter 2)" % nn10[:, -1].mean())
ests = {'MDS': ClassicalMDS(2), 'Isomap': Isomap(n_neighbors=10), 'LLE': LocallyLinearEmbedding(n_neighbors=10, random_state=0),
        'Laplacian': SpectralEmbedding(n_neighbors=10, random_state=0), 't-SNE': TSNE(2, perplexity=30, init='pca', random_state=0)}
Z = {}
def radii(P):                                      # inner and outer radius about the centroid (5th and 95th percentile)
    q = np.linalg.norm(P - P.mean(0), axis=1); return np.percentile(q, 5), np.percentile(q, 95)
print(f"{'method':14s}{'T(10)':>8s}{'Procrustes disparity':>22s}{'inner/outer radius':>20s}")
r0 = radii(X); print(f"{'original':14s}{'':8s}{'':22s}{r0[0] / r0[1]:20.3f}")
for nm, est in ests.items():
    Z[nm] = est.fit_transform(X); r1 = radii(Z[nm])
    print(f"{nm:14s}{trustworthiness(X, Z[nm], n_neighbors=10):8.4f}{procrustes(X, Z[nm])[2]:22.4f}{r1[0] / r1[1]:20.3f}")
# Isomap in detail: pairs on opposite sides of the hole
iso = ests['Isomap']; Dg = iso.dist_matrix_; De = np.linalg.norm(X[:, None] - X[None], axis=2)
i, j = np.unravel_index(np.where(De > 0, De, 0).argmax(), De.shape)
print("farthest pair: straight line %.3f, graph geodesic %.3f" % (De[i, j], Dg[i, j]))
across = (De > 2.5) & (np.abs(np.arctan2(X[:, 1, None], X[:, 0, None]) - np.arctan2(X[None, :, 1], X[None, :, 0])) > 2.8)   # roughly opposite points, far apart
print("opposite pairs (distance > 2.5, angle apart > 2.8 rad): median geodesic/straight ratio %.3f, largest %.3f; all pairs: median %.3f" % (np.median(Dg[across] / De[across]), (Dg[across] / De[across]).max(), np.median(Dg[De > 0] / De[De > 0])))
print("eigenvalues of the classical MDS of the ring (full):", np.round(ClassicalMDS(4).fit(X).eigenvalues_, 2), "  (two dimensions only: the data are flat)")

fig, axs = plt.subplots(1, 6, figsize=(5.4, 1.2), dpi=240)
for ax, (nm, P) in zip(axs, [('original', X)] + list(Z.items())):
    ax.scatter(P[:, 0], P[:, 1], s=1.6, c=np.arctan2(X[:, 1], X[:, 0]), cmap='twilight', lw=0); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(nm, fontsize=8.8, pad=2)
    for s_ in ax.spines.values(): s_.set_visible(False)
fig.subplots_adjust(wspace=0.04, left=0.005, right=0.995, top=0.86, bottom=0.01)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s44c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s44c2.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))
