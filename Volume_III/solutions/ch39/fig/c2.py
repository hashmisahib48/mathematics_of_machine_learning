# Problem 39C.2: why the median preference gives too many exemplars. Writes fig/s39c2.png, prints the numbers.
import sys, os, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning); warnings.simplefilter('ignore', UserWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score as ari

Xb, yb = make_blobs(90, centers=3, cluster_std=0.9, random_state=4)      # the book's data
S = -((Xb[:, None] - Xb[None]) ** 2).sum(-1)
n = 90; iu = np.triu_indices(n, 1); sq = -S[iu]; same = (yb[:, None] == yb[None])[iu]
med = np.median(S)
print("median of all 8100 similarities %.3f  (4005 pairs, each counted twice, plus 90 zeros)" % med)
print("pairs inside a group: %d of %d = %.1f %%; mean squared distance inside %.2f, between %.2f" % (same.sum(), len(sq), 100 * same.mean(), sq[same].mean(), sq[~same].mean()))
print("pairs closer than |median| (squared distance < %.2f): inside a group %.1f %%, between groups %.1f %%" % (-med, 100 * (sq[same] < -med).mean(), 100 * (sq[~same] < -med).mean()))
print("largest squared distance inside a group %.2f, smallest between groups %.2f" % (sq[same].max(), sq[~same].min()))

def net(ex, p):                                    # Eq. 39.4-2 with each sample at its most similar exemplar
    ex = list(ex); rest = [i for i in range(n) if i not in ex]
    return S[np.ix_(rest, ex)].max(1).sum() + p * len(ex)
res = {}
for p in (med, -120.0):
    ap = AffinityPropagation(affinity='precomputed', preference=p, random_state=0, max_iter=400).fit(S)
    res[p] = ap
    ex = ap.cluster_centers_indices_; rest = [i for i in range(n) if i not in ex]
    print("p = %8.3f: %d exemplars, ARI %.3f, mean similarity of a sample to its exemplar %.2f" % (p, len(ex), ari(yb, ap.labels_), S[np.ix_(rest, ex)].max(1).mean()))
for p in (med, -120.0):
    print("  net similarity at p = %.3f:  with the 6-exemplar set %.1f,  with the 3-exemplar set %.1f"
          % (p, net(res[med].cluster_centers_indices_, p), net(res[-120.0].cluster_centers_indices_, p)))
# number of exemplars against the preference
ps = -np.logspace(-1, 2.5, 71); cnt = []
for p in ps:
    cnt.append(len(AffinityPropagation(affinity='precomputed', preference=p, random_state=0, max_iter=400).fit(S).cluster_centers_indices_))
cnt = np.array(cnt); three = ps[cnt == 3]
print("exactly 3 exemplars (grid of 71 values, -0.1 to -316) for p from %.1f to %.1f on the grid" % (three.min(), three.max()))
pm = -np.sqrt(-three.min() * -three.max()); apm = AffinityPropagation(affinity='precomputed', preference=pm, random_state=0, max_iter=400).fit(S)
print("middle of the plateau on a log axis p = %.1f: %d exemplars, ARI %.3f" % (pm, len(apm.cluster_centers_indices_), ari(yb, apm.labels_)))
print("median of the similarities of each sample to its nearest 29 neighbours (a group has 30 points): %.2f" % np.median(np.sort(-S, axis=1)[:, 1:30]))

fig, axs = plt.subplots(1, 2, figsize=(5.4, 1.9), dpi=240)
ax = axs[0]; bins = np.linspace(0, 130, 40)
ax.hist(sq[same], bins=bins, color=BLUE, alpha=0.85, label='inside a group')
ax.hist(sq[~same], bins=bins, color=OR, alpha=0.85, label='between groups')
ax.axvline(-med, color=RED, lw=1.2); ax.text(-med + 2, ax.get_ylim()[1] * 0.9, 'median\n%.1f' % -med, color=RED, fontsize=8, va='top')
ax.set_xlabel('squared distance $-s(i,k)$', fontsize=8.5); ax.set_ylabel('pairs', fontsize=8.5); ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.02, 0.72)); ax.tick_params(labelsize=8)
ax = axs[1]
ax.loglog(-ps, cnt, color=PUR, lw=1.3); ax.axvspan(-three.max(), -three.min(), color=GREEN, alpha=0.2, lw=0)
ax.axvline(-med, color=RED, lw=1.2); ax.set_xlabel('$-p$', fontsize=8.5); ax.set_ylabel('exemplars', fontsize=8.5)
ax.set_yticks([1, 3, 10, 30, 90]); ax.set_yticklabels(['1', '3', '10', '30', '90']); ax.minorticks_off(); ax.tick_params(labelsize=8)
ax.text(np.sqrt(three.min() * three.max()), 1.05, 'three\nexemplars', fontsize=8, color=GREEN, ha='center', va='bottom')
for a in axs: a.grid(color=GRID, lw=0.4)
fig.subplots_adjust(wspace=0.38)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s39c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s39c2.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.6 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.6 / w))
