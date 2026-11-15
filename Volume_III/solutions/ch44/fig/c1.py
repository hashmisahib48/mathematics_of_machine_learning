# Problem 44C.1: placing a new image in a t-SNE map. Writes fig/s44c1.png, verify/c1.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from scipy.optimize import minimize
from scipy.spatial import procrustes
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
Xtr, ytr, Xte, yte = Xtr[:500], ytr[:500], Xte[:200], yte[:200]      # a subset keeps the script under two minutes
Zt = TSNE(2, perplexity=30, init='pca', random_state=0).fit_transform(Xtr)
print("training map: %d images; test images %d" % (len(Xtr), len(Xte)))
knn_map = lambda Zp: KNeighborsClassifier(5).fit(Zt, ytr).score(Zp, yte)
print("5-NN on the 64 pixels (reference): %.4f" % KNeighborsClassifier(5).fit(Xtr, ytr).score(Xte, yte))

# (A) average the map positions of the 5 nearest training images
nn = NearestNeighbors(n_neighbors=5).fit(Xtr); dA, iA = nn.kneighbors(Xte)
ZA = Zt[iA].mean(1)
# (B) minimise the KL divergence of this one point with the training map held fixed
def affinities(dist2, perp=30.0):
    lo, hi, beta = 0.0, np.inf, 1.0
    for _ in range(100):                                       # bisection on beta = 1/(2 sigma^2)
        p = np.exp(-(dist2 - dist2.min()) * beta); p /= p.sum(); H = np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())
        if abs(H - perp) < 1e-5: break
        if H > perp: lo = beta; beta = beta * 2 if hi == np.inf else (beta + hi) / 2
        else: hi = beta; beta = (beta + lo) / 2
    return p
ZB = np.zeros_like(ZA); perps = []
for k, x in enumerate(Xte):
    p = affinities(((Xtr - x) ** 2).sum(1))
    def kl(z):
        w = 1 / (1 + ((Zt - z) ** 2).sum(1)); q = w / w.sum(); m = p > 0
        return (p[m] * np.log(p[m] / q[m])).sum()
    ZB[k] = minimize(kl, ZA[k], method='L-BFGS-B').x
    perps.append(kl(ZA[k]) - kl(ZB[k]))
print("(A) mean of the 5 nearest images:     5-NN accuracy in the map %.4f" % knn_map(ZA))
print("(B) minimise KL, map held fixed:      5-NN accuracy in the map %.4f   (KL lowered on average by %.3f from the start A)" % (knn_map(ZB), np.mean(perps)))
nnz = NearestNeighbors(n_neighbors=2).fit(Zt); spacing = np.median(nnz.kneighbors(Zt)[0][:, 1])
for nm, Zp in (("A", ZA), ("B", ZB)):
    dd = NearestNeighbors(n_neighbors=1).fit(Zt).kneighbors(Zp)[0][:, 0]
    print("(%s) median distance to the nearest training point in the map: %.2f (training spacing %.2f); share of images further than 5 spacings: %.3f" % (nm, np.median(dd), spacing, (dd > 5 * spacing).mean()))
spread = np.array([np.linalg.norm(Zt[i] - Zt[i].mean(0), axis=1).max() for i in iA]); print("(A) spread of the 5 neighbours in the map: median %.2f, share of images whose neighbours are more than 10 spacings apart %.3f" % (np.median(spread), (spread > 10 * spacing).mean()))
# (C) refit everything: the training map changes
Zall = TSNE(2, perplexity=30, init='pca', random_state=0).fit_transform(np.vstack([Xtr, Xte]))
print("(C) refit on training + test images: Procrustes disparity of the training part against the first map %.4f" % procrustes(Zt, Zall[:len(Xtr)])[2])
print("    5-NN accuracy of the test images in the joint map, labels of the training images only: %.4f" % KNeighborsClassifier(5).fit(Zall[:len(Xtr)], ytr).score(Zall[len(Xtr):], yte))

cols = ['#1F5FBF', '#F39C12', '#2E8B57', '#C0392B', '#7B3FA0', '#0E7C86', '#D6336C', '#8C6D1F', '#555555', '#6AA84F']
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.5), dpi=240)
for ax, Zp, ttl in zip(axs, (ZA, ZB), ('(A) mean of 5 neighbours', '(B) minimise KL, map fixed')):
    ax.scatter(Zt[:, 0], Zt[:, 1], s=2.5, color='#D9DDE3', lw=0)
    ax.scatter(Zp[:, 0], Zp[:, 1], s=4, c=[cols[k] for k in yte], lw=0)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl, fontsize=8.8, pad=2)
    for s in ax.spines.values(): s.set_visible(False)
fig.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.92, bottom=0.01)
fs = min(t_.get_fontsize() for t_ in fig.findobj(mtext.Text) if t_.get_text().strip() and t_.get_visible())
fig.savefig('fig/s44c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s44c1.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))
