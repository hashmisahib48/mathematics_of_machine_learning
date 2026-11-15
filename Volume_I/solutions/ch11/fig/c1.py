# Problem 11C.1: stability of the lasso selection under the bootstrap. Writes fig/s11c1.png
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

d = load_diabetes(); names = d.feature_names
Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=0.25, random_state=0)
rng = np.random.default_rng(0); B = 60; a = 0.0787
W = np.zeros((B, 10))
for b in range(B):
    idx = rng.integers(0, len(ytr), len(ytr))
    W[b] = Lasso(alpha=a, max_iter=100000, tol=1e-7).fit(Xtr[idx], ytr[idx]).coef_
S = W != 0; freq = S.mean(0)
print("fraction of the %d bootstrap fits in which each feature is non-zero (alpha = %.4f):" % (B, a))
for nm, f, k in sorted(zip(names, freq, S.sum(0)), key=lambda t: -t[1]): print(f"  {nm:4s} {f:.2f}  ({k} of {B})")
print("binomial standard error of a frequency f with B = 60: sqrt(f(1-f)/B) at f = 0.5: %.3f, at f = 0.7: %.3f" % (np.sqrt(.25 / B), np.sqrt(.21 / B)))
print("share of distinct patients in a bootstrap sample of n = %d: %.3f" % (len(ytr), 1 - (1 - 1 / len(ytr)) ** len(ytr)))
print("size of the selected set: min %d, max %d, mean %.2f" % (S.sum(1).min(), S.sum(1).max(), S.sum(1).mean()))
i1, i2 = names.index('s1'), names.index('s2')
both, only1, only2, none = (S[:, i1] & S[:, i2]).sum(), (S[:, i1] & ~S[:, i2]).sum(), (~S[:, i1] & S[:, i2]).sum(), (~S[:, i1] & ~S[:, i2]).sum()
print("s1 and s2: both %d, only s1 %d, only s2 %d, neither %d" % (both, only1, only2, none))
print("correlation of the weights of s1 and s2 over the fits: %.3f ; correlation of the two columns in the data: %.3f" % (np.corrcoef(W[:, i1], W[:, i2])[0, 1], np.corrcoef(Xtr[:, i1], Xtr[:, i2])[0, 1]))
full = Lasso(alpha=a, max_iter=100000, tol=1e-7).fit(Xtr, ytr)
print("fit on the original training set selects:", [n for n, w in zip(names, full.coef_) if w != 0])
print("selected in at least 90 per cent of the fits:", [n for n, f in zip(names, freq) if f >= 0.9])

fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.95), dpi=240, gridspec_kw={'width_ratios': [1.25, 1]})
ax = axs[0]; order = np.argsort(-freq)
ax.bar(range(10), freq[order], color=[MAG if names[j] in ('s1', 's2') else GREEN for j in order], width=0.7)
ax.set_xticks(range(10)); ax.set_xticklabels([names[j] for j in order], fontsize=8.5); ax.axhline(0.9, color=GREY, lw=0.8, ls='--')
ax.set_ylabel('fraction of fits', fontsize=8.5); ax.set_ylim(0, 1.1); ax.grid(color=GRID, lw=0.5, axis='y'); ax.set_axisbelow(True); ax.tick_params(labelsize=8.5)
ax = axs[1]; ax.scatter(W[:, i1], W[:, i2], s=9, color=BLUE, lw=0); ax.axhline(0, color=GREY, lw=0.5); ax.axvline(0, color=GREY, lw=0.5)
ax.set_xlabel('weight of s1', fontsize=8.5); ax.set_ylabel('weight of s2', fontsize=8.5); ax.tick_params(labelsize=8.5); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(pad=0.4, w_pad=0.8)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s11c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s11c1.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))
