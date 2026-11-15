# Problem 20C.2: which feature decides the wine distances. Writes fig/s20c2.png, verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

data = load_wine(); X = data.data; names = data.feature_names; n, p = X.shape
Z = StandardScaler().fit_transform(X)
def exact(M):                 # mean over ALL pairs i != j of (x_ij - x_kj)^2, per feature, as a share of the total
    s = 2 * n / (n - 1) * M.var(0); return s / s.sum()
def sampled(M, pairs=3000):   # the book's figure: 3000 random pairs
    r = np.random.default_rng(0); i, j = r.integers(0, len(M), (2, pairs)); keep = i != j
    s = ((M[i[keep]] - M[j[keep]]) ** 2).sum(0); return s / s.sum()
er, es, sr, ss = exact(X), exact(Z), sampled(X), sampled(Z)
print("feature with the largest raw share:", names[er.argmax()], "(column %d)" % (er.argmax() + 1))
print("raw shares, all pairs: largest %.4f, second %.4f (%s), the other eleven together %.4f"
      % (er.max(), np.sort(er)[-2], names[np.argsort(er)[-2]], 1 - er.max() - np.sort(er)[-2]))
print("raw share from the 3000 random pairs of the book's figure: %.4f" % sr.max())
print("variances: proline %.1f, total of all 13 %.1f" % (X[:, 12].var(), X.var(0).sum()))
print("standardised shares, all pairs: min %.4f, max %.4f (1/13 = %.4f)" % (es.min(), es.max(), 1 / 13))
print("standardised shares, 3000 random pairs: min %.4f, max %.4f; largest column %d" % (ss.min(), ss.max(), ss.argmax() + 1))
rng = np.random.default_rng(1); sd = [sampled(Z, 3000)]
for s in range(1, 6):
    r = np.random.default_rng(s); i, j = r.integers(0, n, (2, 3000)); keep = i != j; v = ((Z[i[keep]] - Z[j[keep]]) ** 2).sum(0); sd.append(v / v.sum())
print("largest standardised share in six different random samples of 3000 pairs:", [round(float(v.max()), 4) for v in sd])
# what it means for the neighbours
k = 5
nr = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
npro = NearestNeighbors(n_neighbors=k + 1).fit(X[:, [12]]).kneighbors(X[:, [12]], return_distance=False)[:, 1:]
ns_ = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
ov = lambda a, b: np.mean([len(set(u) & set(v)) / k for u, v in zip(a, b)])
print("mean share of the 5 neighbours that raw distance and proline alone agree on: %.3f" % ov(nr, npro))
print("mean share of the 5 neighbours shared by raw and standardised distance:     %.3f" % ov(nr, ns_))
y = data.target
print("share of the 5 raw neighbours that have the wine's own cultivar: %.3f; standardised: %.3f; proline alone: %.3f"
      % (np.mean(y[nr] == y[:, None]), np.mean(y[ns_] == y[:, None]), np.mean(y[npro] == y[:, None])))
print("proline range of the three cultivars: " + "; ".join("%d: %.0f to %.0f (mean %.0f)" % (c, X[y == c, 12].min(), X[y == c, 12].max(), X[y == c, 12].mean()) for c in range(3)))

fig, ax = plt.subplots(figsize=(4.6, 1.6), dpi=240)
ax.bar(np.arange(p) - 0.2, er, width=0.4, color=RED, label='raw'); ax.bar(np.arange(p) + 0.2, es, width=0.4, color=GREEN, label='standardised')
ax.set_xticks(range(p)); ax.set_xticklabels([str(i + 1) for i in range(p)]); ax.set_xlabel('feature number (13 = proline)', fontsize=9); ax.set_ylabel('share of $d^2$', fontsize=9)
ax.axhline(1 / 13, color=GREY, lw=0.8, ls='--'); ax.text(0.2, 1 / 13 + 0.03, '1/13', fontsize=8.5, color=GREY)
ax.legend(fontsize=8.5, loc='center left'); ax.tick_params(labelsize=8.5); ax.grid(color=GRID, lw=0.5, axis='y'); ax.set_axisbelow(True)
fig.tight_layout()
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s20c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s20c2.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.3 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.3 / w))
