# Problem 42C.2: the dominant feature of the wine data. Writes fig/s42c2.png, verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

wine = load_wine(); X, y = wine.data, wine.target; names = wine.feature_names; ip = names.index('proline'); im_ = names.index('magnesium')
v = X.var(0, ddof=1); o = np.argsort(v)[::-1]
print("variances (ddof = 1), largest first:", [(names[i], round(float(v[i]), 4)) for i in o[:4]], " smallest:", (names[o[-1]], round(float(v[o[-1]]), 4)))
print("ratio of the largest to the smallest variance: %.2e ; total variance %.2f ; share of proline %.4f" % (v.max() / v.min(), v.sum(), v[ip] / v.sum()))
p = PCA().fit(X); print("raw PCA: ratios of the first 4 components", p.explained_variance_ratio_[:4].round(4).tolist(), " PC1 weights: proline %.4f, magnesium %.4f" % tuple(p.components_[0][[ip, im_]]))
acc = lambda Z: cross_val_score(KNeighborsClassifier(5), Z, y, cv=5).mean()
print("5-NN, 5-fold, on the first two raw scores: %.4f ; on the raw proline alone: %.4f" % (acc(p.transform(X)[:, :2]), acc(X[:, [ip]])))
print("\nthe same PCA after changing the unit of proline only:")
variants = {'raw': X.copy()}
Xg = X.copy(); Xg[:, ip] /= 1000; variants['proline in g/l'] = Xg           # mg/l -> g/l
Xs = StandardScaler().fit_transform(X); variants['standardised'] = Xs
cum = {}
for nm, D in variants.items():
    q = PCA().fit(D); r = q.explained_variance_ratio_; cum[nm] = np.cumsum(r); top = np.argmax(np.abs(q.components_[0]))
    print("%-16s ratios %s  PC1 is mainly %s (weight %.2f)  5-NN on 2 scores %.4f" % (nm, r[:4].round(3).tolist(), names[top], abs(q.components_[0][top]), acc(q.transform(D)[:, :2])))
print("\nstandardised: cumulative shares", cum['standardised'][:8].round(3).tolist(), " components for 80, 90, 95 per cent:", [int(np.argmax(cum['standardised'] > f)) + 1 for f in (0.8, 0.9, 0.95)])
r = PCA().fit(Xs).explained_variance_; print("participation ratio (sum lambda)^2 / sum lambda^2 of the standardised data: %.2f of 13 ; of the raw data: %.2f" % (r.sum() ** 2 / (r ** 2).sum(), PCA().fit(X).explained_variance_.sum() ** 2 / (PCA().fit(X).explained_variance_ ** 2).sum()))
print("5-NN on all 13 raw features: %.4f ; on all 13 standardised: %.4f" % (acc(X), acc(Xs)))

fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.9), dpi=240)
a_ = axs[0]; ks = np.arange(1, 9); col = {'raw': RED, 'proline in g/l': OR, 'standardised': BLUE}; mk = {'raw': 'o', 'proline in g/l': 's', 'standardised': '^'}
for nm, c in cum.items(): a_.plot(ks, c[:8], '-' + mk[nm], color=col[nm], ms=3, lw=1.1, label=nm)
a_.set_xlabel('components $k$'); a_.set_ylabel('cumulative share'); a_.set_ylim(0.3, 1.03); a_.legend(fontsize=8.0, loc='lower right', borderaxespad=0.2, handlelength=1.4)
a_ = axs[1]; srt = np.sort(v)[::-1]; a_.bar(range(13), srt, color=[RED] + [BLUE] * 12, width=0.75); a_.set_yscale('log'); a_.set_xticks([]); a_.set_xlabel('13 features, by variance'); a_.set_ylabel('variance')
a_.text(0.6, srt[0] * 0.5, 'proline', fontsize=8.2, color=RED, va='top'); a_.set_ylim(1e-3, 3e5)
for a_ in axs: a_.grid(color=GRID, lw=0.5); a_.set_axisbelow(True); a_.tick_params(labelsize=8.3); a_.xaxis.label.set_size(8.6); a_.yaxis.label.set_size(8.6)
fig.subplots_adjust(wspace=0.35)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s42c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s42c2.png') as im: w_ = im.size[0] / 240
print("\nfigure width %.2f in, printed 5.0 in, smallest font %.1f pt -> printed %.1f pt" % (w_, fs, fs * 5.0 / w_))
