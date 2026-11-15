# Question 2: two whorls of the noisy swiss roll and the neighbourhoods of one point. Writes fig/s44q2.png, verify/q2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors

Xn, _ = make_swiss_roll(1000, noise=0.5, random_state=0); _, t = make_swiss_roll(1000, noise=0.0, random_state=0)
d, idx = NearestNeighbors(n_neighbors=41).fit(Xn).kneighbors(Xn)
far = np.abs(t[idx] - t[:, None]) > np.pi
cnt20 = far[:, 1:21].sum(1); i = int(cnt20.argmax())
print("point %d: %d of its 20 nearest neighbours lie on another whorl; among its 10 nearest: %d" % (i, cnt20[i], far[i, 1:11].sum()))
print("distance to its 10th neighbour %.2f, to its 20th %.2f; to the nearest point of the other whorl %.2f" % (d[i, 10], d[i, 20], d[i][far[i]].min()))
for m in (5, 10, 20, 40):
    print("m = %2d: share of edges between different whorls %.4f, share of points with such an edge %.3f" % (m, far[:, 1:m + 1].mean(), far[:, 1:m + 1].any(1).mean()))

fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=240)
ax.scatter(Xn[:, 0], Xn[:, 2], s=3, color='#C9CED6', lw=0)
n10 = idx[i, 1:11]; n20 = idx[i, 11:21]
ax.scatter(Xn[n20, 0], Xn[n20, 2], s=11, color=OR, lw=0, label='neighbours 11 to 20')
ax.scatter(Xn[n10, 0], Xn[n10, 2], s=11, color=BLUE, lw=0, label='10 nearest neighbours')
bad = idx[i, 1:21][far[i, 1:21]]; ax.scatter(Xn[bad, 0], Xn[bad, 2], s=34, facecolor='none', edgecolor=RED, lw=1.0, label='on another whorl')
ax.scatter(Xn[i, 0], Xn[i, 2], marker='*', s=70, color=NAVY, zorder=5)
for r, ls in ((d[i, 10], '-'), (d[i, 20], '--')): ax.add_patch(plt.Circle((Xn[i, 0], Xn[i, 2]), r, fill=False, color=GREY, lw=0.7, ls=ls))
ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)
ax.legend(loc='upper left', fontsize=7.8, handlelength=1.0, borderaxespad=0.0, labelspacing=0.2, bbox_to_anchor=(0.98, 1.0))
fig.subplots_adjust(left=0.01, right=0.55, top=0.99, bottom=0.01)
fs = min(t_.get_fontsize() for t_ in fig.findobj(mtext.Text) if t_.get_text().strip() and t_.get_visible())
fig.savefig('fig/s44q2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s44q2.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 3.4 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 3.4 / w))
