# Problem 59C.3: segments of customers, with and without real groups. Writes fig/s59c3.png, prints verify/c3.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import adjusted_rand_score as ari, silhouette_score
from sklearn.preprocessing import StandardScaler

n, p = 5000, 40
def make(structured, seed=0):
    """Invented shop: 40 count/amount columns driven by 3 hidden tastes; the tastes are either
    spread continuously or sit in five tight types."""
    r = np.random.RandomState(seed)
    if structured:
        C = 2.2 * np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1.4]]); z = C[r.randint(0, 5, n)] + r.normal(0, 0.45, (n, 3))
    else:
        z = r.normal(0, 1, (n, 3))
    L = r.normal(0, 0.45, (3, p))
    X = np.log1p(r.poisson(np.exp(0.8 + z @ L + r.normal(0, 0.25, (n, p)))))
    y = np.exp(0.5 * z[:, 0] - 0.4 * z[:, 1] + r.normal(0, 0.5, n))          # next-quarter spend
    return StandardScaler().fit_transform(X), y

Ks = list(range(2, 9)); out = {}
for name, st in (('continuous', False), ('five types', True)):
    X, y = make(st)
    half = np.arange(n) % 2 == 0
    J = [KMeans(k, n_init=3, random_state=0).fit(X).inertia_ for k in [1] + Ks]
    sil, stab = [], []
    for k in Ks:
        a = KMeans(k, n_init=3, random_state=0).fit(X[half]); b = KMeans(k, n_init=3, random_state=0).fit(X[~half])
        stab.append(ari(a.predict(X), b.predict(X)))                 # two halves, same customers labelled
        sil.append(silhouette_score(X, KMeans(k, n_init=3, random_state=0).fit_predict(X), sample_size=1500, random_state=0))
    out[name] = (J, sil, stab)
    print("== %s" % name)
    print(" K   J(K)/J(1)  decrease  silhouette  stability(ARI of the two halves)")
    for i, k in enumerate(Ks):
        print("%2d  %9.3f %9.0f %10.3f %10.3f" % (k, J[i + 1] / J[0], J[i] - J[i + 1], sil[i], stab[i]))
    # meaning: do the segments tell more about next-quarter spend than a simple summary does?  fit on one half, score on the other
    seg = KMeans(5, n_init=3, random_state=0).fit(X[half]); oh = np.eye(5)[seg.predict(X)]
    pc = PCA(3, random_state=0).fit(X[half]).transform(X)
    for lab, F in (('five segments (one-hot)', oh), ('three principal components', pc)):
        m = LinearRegression().fit(F[half], np.log(y[half])); print("  held-out R2 for log spend, %-28s %.3f" % (lab, m.score(F[~half], np.log(y[~half]))))
    print("  variance kept by 3 components %.3f" % PCA(3, random_state=0).fit(X).explained_variance_ratio_.sum())

fig, axs = plt.subplots(1, 3, figsize=(4.9, 1.7), dpi=240)
for nm, col, mk in (('continuous', RED, 'o'), ('five types', GREEN, 's')):
    J, sil, stab = out[nm]
    axs[0].plot([1] + Ks, np.array(J) / J[0], color=col, marker=mk, ms=2.8, lw=1.1, label=nm)
    axs[1].plot(Ks, sil, color=col, marker=mk, ms=2.8, lw=1.1)
    axs[2].plot(Ks, stab, color=col, marker=mk, ms=2.8, lw=1.1)
for ax, t in zip(axs, ('inertia, share of J(1)', 'silhouette', 'stability (ARI)')):
    ax.set_title(t, fontsize=8, pad=3); ax.set_xlabel('K', fontsize=8); ax.tick_params(labelsize=8); ax.set_xticks([2, 4, 6, 8]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
axs[0].set_xticks([1, 3, 5, 7]); axs[0].legend(fontsize=8, loc='upper right', handlelength=1.2, borderaxespad=0.1)
fig.tight_layout(pad=0.4, w_pad=0.6)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s59c3.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s59c3.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))
