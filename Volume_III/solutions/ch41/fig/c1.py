# Problem 41C.1: why the raw clustering of the wines has the better silhouette. Writes fig/s41c1.png, verify/c1.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn import metrics as m
from sklearn.preprocessing import StandardScaler

wine = load_wine(); X, y = wine.data, wine.target; names = list(wine.feature_names); ip = names.index('proline'); iflav = names.index('flavanoids')
Xs = StandardScaler().fit_transform(X)
lr = KMeans(3, n_init=10, random_state=0).fit_predict(X); ls = KMeans(3, n_init=10, random_state=0).fit_predict(Xs)
v = X.var(0); print("variance share of proline: %.4f ; next largest (%s): %.4f" % (v[ip] / v.sum(), names[np.argsort(v)[-2]], np.sort(v)[-2] / v.sum()))
d_all, d_pro = pdist(X), pdist(X[:, [ip]])
print("correlation of the 15753 pairwise distances, all 13 features against proline alone: %.5f" % np.corrcoef(d_all, d_pro)[0, 1])
print("sum of squared distances carried by proline: %.4f" % ((d_pro ** 2).sum() / (d_all ** 2).sum()))
lp = KMeans(3, n_init=10, random_state=0).fit_predict(X[:, [ip]])
print("k-means on proline alone: ARI with the raw 13-feature clustering %.3f ; silhouette %.3f ; ARI with cultivars %.3f" % (m.adjusted_rand_score(lr, lp), m.silhouette_score(X[:, [ip]], lp), m.adjusted_rand_score(y, lp)))
print("raw clustering: silhouette %.3f, ARI %.3f   standardised: silhouette %.3f, ARI %.3f" % (m.silhouette_score(X, lr), m.adjusted_rand_score(y, lr), m.silhouette_score(Xs, ls), m.adjusted_rand_score(y, ls)))
print("\nproline range of each raw cluster (cluster, n, min, max):")
for k in np.argsort([X[lr == k, ip].mean() for k in range(3)]): print("  cluster %d: n = %3d  %6.0f to %6.0f" % (k, (lr == k).sum(), X[lr == k, ip].min(), X[lr == k, ip].max()))
print("proline of each cultivar (mean, sd, min, max):")
for c in range(3): print("  cultivar %d: %6.0f %6.0f %6.0f %6.0f" % (c + 1, X[y == c, ip].mean(), X[y == c, ip].std(ddof=1), X[y == c, ip].min(), X[y == c, ip].max()))
print("contingency table, rows cultivars, columns raw clusters ordered by proline:")
order = np.argsort([X[lr == k, ip].mean() for k in range(3)]); print(m.cluster.contingency_matrix(y, lr)[:, order])
print("same for the standardised clustering:"); print(m.cluster.contingency_matrix(y, ls))
a, b = X[lr == order[0], ip].max(), X[lr == order[1], ip].min(); print("gap between the lowest two raw clusters in proline: %.0f to %.0f" % (a, b))
print("silhouette of the cultivars: raw space %.3f, standardised space %.3f" % (m.silhouette_score(X, y), m.silhouette_score(Xs, y)))
# which feature pairs: flavanoids separates cultivar 3 from 2, proline does not
print("flavanoids mean by cultivar:", [round(float(X[y == c, iflav].mean()), 2) for c in range(3)])

fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.9), dpi=240, sharey=True)
mk = ['o', 's', '^']; col = [BLUE, OR, TEAL]
for ax, lab, ttl in ((axs[0], np.argsort(order)[lr], 'raw k-means, K = 3'), (axs[1], y, 'cultivars')):
    for c in range(3):
        mm = lab == c; ax.scatter(X[mm, ip], X[mm, iflav], s=7, color=col[c], marker=mk[c], lw=0, label=(['low proline', 'middle', 'high proline'] if ttl[0] == 'r' else ['cultivar 1', 'cultivar 2', 'cultivar 3'])[c])
    ax.set_xlabel('proline'); ax.set_title(ttl, fontsize=8.6); ax.legend(fontsize=7.8, loc='upper right', handletextpad=0.1, borderaxespad=0.1, labelspacing=0.2, markerscale=1.2)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(labelsize=8.3); ax.xaxis.label.set_size(8.6)
    ax.set_xlim(250, 1800)
axs[0].set_ylabel('flavanoids'); axs[0].yaxis.label.set_size(8.6); axs[0].set_ylim(0, 6.2)
fig.subplots_adjust(wspace=0.08)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s41c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s41c1.png') as im: w = im.size[0] / 240
print("\nfigure width %.2f in, printed 5.0 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 5.0 / w))
