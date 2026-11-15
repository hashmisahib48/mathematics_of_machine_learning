# Problem 36C.2: a curve with no elbow. Writes fig/s36c2.png, verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, adjusted_rand_score

U = np.random.default_rng(0).uniform(size=(400, 2))                   # the book's data
cent = [(-5, -3), (-3, 4), (4, -4), (5, 3)]                           # the four blobs of Section 36.11
B, _ = make_blobs(400, centers=cent, cluster_std=1.5, random_state=3)
Ks = range(1, 9)
def curve(X): return np.array([KMeans(k, n_init=5, random_state=0).fit(X).inertia_ for k in Ks])
JU, JB = curve(U), curve(B)
print("K        :", list(Ks)); print("J uniform:", JU.round(1).tolist()); print("J blobs  :", JB.round(0).tolist())
print("J/J(1) uniform:", (JU / JU[0]).round(3).tolist()); print("J/J(1) blobs  :", (JB / JB[0]).round(3).tolist())
print("drops uniform:", (-np.diff(JU)).round(1).tolist()); print("drops blobs  :", (-np.diff(JB)).round(0).tolist())
print("ratio of successive drops, uniform:", (np.diff(JU)[1:] / np.diff(JU)[:-1]).round(2).tolist())
print("ratio of successive drops, blobs  :", (np.diff(JB)[1:] / np.diff(JB)[:-1]).round(2).tolist())
print("K * J(K) uniform:", (np.array(list(Ks)) * JU).round(1).tolist())
hexc = 5 / (18 * np.sqrt(3)); print("theory for a square of area 1, many cells of hexagonal shape: J = n*%.4f/K ; n/6 at K=1 = %.1f" % (hexc, 400 / 6))
print("theory K=8: %.2f   measured %.2f" % (400 * hexc / 8, JU[7]))
def stab(X, k, reps=10):
    r = np.random.default_rng(1); out = []
    for _ in range(reps):
        p = r.permutation(len(X)); a, b = p[:len(X) // 2], p[len(X) // 2:]
        la = KMeans(k, n_init=5, random_state=0).fit(X[a]).predict(X); lb = KMeans(k, n_init=5, random_state=0).fit(X[b]).predict(X)
        out.append(adjusted_rand_score(la, lb))
    return np.mean(out)
SU = [silhouette_score(U, KMeans(k, n_init=5, random_state=0).fit(U).labels_) for k in range(2, 9)]
SB = [silhouette_score(B, KMeans(k, n_init=5, random_state=0).fit(B).labels_) for k in range(2, 9)]
TU = [stab(U, k) for k in range(2, 9)]; TB = [stab(B, k) for k in range(2, 9)]
print("K          :", list(range(2, 9)))
print("silhouette uniform:", np.round(SU, 3).tolist()); print("silhouette blobs  :", np.round(SB, 3).tolist())
print("stability ARI uniform:", np.round(TU, 3).tolist()); print("stability ARI blobs  :", np.round(TB, 3).tolist())

fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.9), dpi=240); K1 = np.array(list(Ks))
a = axs[0]; a.plot(K1, JB / JB[0], '-o', color=BLUE, ms=3, lw=1.1, label='four blobs'); a.plot(K1, JU / JU[0], '-s', color=RED, ms=3, lw=1.1, label='uniform square')
a.plot(K1, 1 / K1, ls='--', color=GREY, lw=0.8, label='1/K'); a.set_xlabel('K'); a.set_ylabel('J(K) / J(1)'); a.legend(fontsize=8.2, loc='upper right'); a.set_ylim(0, 1.05)
a = axs[1]; K2 = np.arange(2, 9); a.plot(K2, SB, '-o', color=BLUE, ms=3, lw=1.1, label='four blobs'); a.plot(K2, SU, '-s', color=RED, ms=3, lw=1.1, label='uniform square')
a.set_xlabel('K'); a.set_ylabel('mean silhouette'); a.set_ylim(0, 0.8); a.legend(fontsize=8.2, loc='upper right')
for a in axs: a.grid(color=GRID, lw=0.5); a.set_axisbelow(True); a.tick_params(labelsize=8.3); a.xaxis.label.set_size(8.6); a.yaxis.label.set_size(8.6)
fig.subplots_adjust(wspace=0.32)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s36c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s36c2.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))
