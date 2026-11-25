import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from figstyle import *
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine, load_sample_image, make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
P = lambda n: os.path.join(HERE, 'png', n)
def fin(fig, n): fig.tight_layout(pad=0.3); fig.savefig(P(n), dpi=300, facecolor='white'); plt.close(fig)
def sty(ax, xl, yl): ax.set_xlabel(xl, fontsize=7.5); ax.set_ylabel(yl, fontsize=7.5); ax.tick_params(labelsize=6.8, length=2.2)
# 10.4-1 example
X = np.array([[1, 1], [2, 1], [1, 2], [5, 4], [6, 5], [7, 4.]])
C = [np.array([[1, 1], [2, 1.]]), np.array([[1, 1.5], [5, 3.5]]), np.array([[4 / 3, 4 / 3], [6, 13 / 3]])]
fig, ax = plt.subplots(figsize=(1.75, 1.62), dpi=300)
ax.plot(X[:3, 0], X[:3, 1], 'o', ms=3.5, color=BLUE); ax.plot(X[3:, 0], X[3:, 1], 's', ms=3.5, color=OR)
for j, col in [(0, BLUE), (1, OR)]:
    path = np.array([c[j] for c in C]); ax.plot(path[:, 0], path[:, 1], '-', color=col, lw=0.7, alpha=0.8)
    ax.plot(path[0, 0], path[0, 1], 'o', ms=8, mfc='none', mec=col, mew=0.9); ax.plot(path[1:-1, 0], path[1:-1, 1], 'x', ms=4.5, color=col, mew=1.0)
    ax.plot(path[-1, 0], path[-1, 1], '*', ms=8, color=col, mec='k', mew=0.3)
xx = np.linspace(0, 8, 2); m = (C[2][0] + C[2][1]) / 2; d = C[2][1] - C[2][0]
ax.plot(xx, m[1] - d[0] / d[1] * (xx - m[0]), color=GREY, lw=0.6, ls='--')
ax.set_xlim(0, 8); ax.set_ylim(0, 6); ax.set_aspect('equal'); sty(ax, '$x_1$', '$x_2$'); fin(fig, 'f101.png')
# 10.2-1 failure on elongated clusters
Xb, yb = make_blobs(n_samples=300, centers=[[0, 0], [0, 3.2]], cluster_std=[[2.2, 0.45], [2.2, 0.45]], random_state=1)
lab = KMeans(2, n_init=10, random_state=0).fit_predict(Xb)
fig, ax = plt.subplots(figsize=(1.75, 1.45), dpi=300)
for k, col, mk in [(0, BLUE, 'o'), (1, OR, 's')]: ax.plot(Xb[lab == k, 0], Xb[lab == k, 1], mk, ms=1.6, color=col, alpha=0.8)
ax.set_aspect('equal'); sty(ax, '$x_1$', '$x_2$'); fin(fig, 'f102.png')
# 10.7-1 wine k choice
Z = StandardScaler().fit_transform(load_wine().data); ks = range(2, 8); J = []; S = []
for k in ks: m = KMeans(k, n_init=10, random_state=0).fit(Z); J.append(m.inertia_); S.append(silhouette_score(Z, m.labels_))
fig, ax = plt.subplots(figsize=(1.75, 1.6), dpi=300); ax2 = ax.twinx()
ax.plot(ks, J, 'o-', ms=2.5, color=BLUE, lw=0.9); ax2.plot(ks, S, 's-', ms=2.5, color=RED, lw=0.9)
ax.axvline(3, color=GREY, lw=0.5, ls='--'); ax.set_xticks(list(ks)); sty(ax, 'number of clusters $K$', 'inertia'); ax.yaxis.label.set_color(BLUE)
ax2.set_ylabel('silhouette', fontsize=7.5, color=RED); ax2.tick_params(labelsize=6.8, length=2.2); ax2.spines['right'].set_visible(True); fin(fig, 'f103.png')
# 10.7-2 colors
img = load_sample_image('china.jpg') / 255.0; Pp = img.reshape(-1, 3); rng = np.random.default_rng(0)
m = KMeans(16, n_init=4, random_state=0).fit(Pp[rng.choice(len(Pp), 2000, replace=False)]); Q = m.cluster_centers_[m.predict(Pp)].reshape(img.shape)
fig, axs = plt.subplots(2, 1, figsize=(1.75, 2.55), dpi=300)
for ax, im, t in zip(axs, [img, Q], ['original, 96 615 colors', '16 colors']): ax.imshow(im); ax.axis('off'); ax.set_title(t, fontsize=7, pad=2)
fin(fig, 'f104.png')
# P10-2
fig, ax = plt.subplots(figsize=(1.75, 1.2), dpi=300); ax.imshow(load_sample_image('flower.jpg')); ax.axis('off'); fin(fig, 'pj102.png')
# P10-1 wines by two features, colored by cultivar (true labels, not used by k-means)
W = load_wine(); names = list(W.feature_names); i, j = names.index('alcohol'), names.index('flavanoids')
fig, ax = plt.subplots(figsize=(1.75, 1.55), dpi=300)
for c, col, mk in [(0, BLUE, 'o'), (1, OR, 's'), (2, GREEN, '^')]:
    ax.plot(W.data[W.target == c, i], W.data[W.target == c, j], mk, ms=2.2, color=col, alpha=0.85, label=f'{c + 1}')
ax.legend(title='cultivar', title_fontsize=6.5, fontsize=6.5, loc='upper center', ncol=3, handletextpad=0.1, columnspacing=0.8, borderaxespad=0.1); sty(ax, 'alcohol (%)', 'flavanoids'); ax.set_ylim(0, 7.6); ax.set_yticks([0, 2, 4, 6]); fin(fig, 'pj101.png')
# P10-3 one example of each digit
from sklearn.datasets import load_digits
Dg = load_digits(); fig, axs = plt.subplots(2, 5, figsize=(1.75, 0.82), dpi=300)
for d, ax in enumerate(axs.ravel()): ax.imshow(Dg.images[np.where(Dg.target == d)[0][0]], cmap='gray_r'); ax.axis('off')
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.08, hspace=0.08); fig.savefig(P('pj103.png'), dpi=300, facecolor='white'); plt.close(fig)
# P10-4 the days of the ammonia plant by air flow, with the two start centers of part (a)
af = [50] * 5 + [56] + [58] * 6 + [62] * 5 + [70] + [75] + [80] * 2
fig, ax = plt.subplots(figsize=(1.75, 1.15), dpi=300)
for v in sorted(set(af)): ax.plot([v] * af.count(v), range(1, af.count(v) + 1), 'o', ms=3.6, color=BLUE)
for v, col in [(50, RED), (80, GREEN)]: ax.plot([v], [-0.7], '^', ms=5, color=col, clip_on=False)
ax.set_ylim(-1.2, 7); ax.set_xlim(46, 84); ax.set_yticks([2, 4, 6]); sty(ax, 'air flow', 'days'); fin(fig, 'pj104.png')
print('ok')
