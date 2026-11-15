# Problem 42C.1: whitening and the constant pixels of the digits. Writes fig/s42c1.png, verify/c1.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X, y = load_digits(return_X_y=True); Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
const = np.where(Xtr.std(0) == 0)[0]; print("constant pixels in the training images:", const.tolist())
print("test images with a non-zero grey value in one of them: %d of %d ; their values:" % ((Xte[:, const] != 0).any(1).sum(), len(Xte)), np.unique(Xte[:, const][(Xte[:, const] != 0)]).tolist())
p = PCA(64).fit(Xtr); lam = p.explained_variance_; print("solver chosen by auto:", p._fit_svd_solver)
print("eigenvalues 56 to 64:", ["%.2e" % v for v in lam[55:]]); eps = np.finfo(float).eps
print("explained_variance_ below eps^2 (scale clipped to eps):", int((np.sqrt(np.maximum(lam, 0)) < eps).sum()), " negative:", int((lam < 0).sum()))
print("scale sqrt(lambda): PC1 %.2f, PC20 %.2f, PC40 %.3f, PC60 %.4f, and for the four zero components the clip value eps = %.2e" % (np.sqrt(lam[0]), np.sqrt(lam[19]), np.sqrt(lam[39]), np.sqrt(lam[59]), eps))
print("weight 1/scale of PC60 relative to PC1: %.0f ; of a zero component relative to PC1: %.1e" % (np.sqrt(lam[0] / lam[59]), np.sqrt(lam[0]) / eps))
print("share of the variance in components 41-60: %.4f ; share of the squared distances in the whitened space (20 of the 60 components): %.3f" % (lam[40:60].sum() / lam.sum(), 20 / 60))
print("share of the whitened variance in the first 20 components: %.3f (they hold %.3f of the raw variance)" % (20 / 60, lam[:20].sum() / lam.sum()))
Zte = PCA(64, whiten=True).fit(Xtr).transform(Xte); print("largest |whitened test score| with 64 components (default solver): %.2e ; with 60: %.1f" % (np.abs(Zte).max(), np.abs(PCA(60, whiten=True).fit(Xtr).transform(Xte)).max()))
Zf = PCA(64, svd_solver="full", whiten=True).fit(Xtr).transform(Xte); print("with svd_solver='full': largest %.2e, 5-NN test accuracy %.4f" % (np.abs(Zf).max(), KNeighborsClassifier(5).fit(PCA(64, svd_solver="full", whiten=True).fit(Xtr).transform(Xtr), ytr).score(Zf, yte)))
print("median |whitened test score| on the four last components (default solver):", ["%.1e" % v for v in np.median(np.abs(Zte[:, 60:]), 0)])
print("whitened scores on the four zero components: largest |score| per column", np.abs(Zte[:, 60:]).max(0).round(0).tolist() if False else ["%.1e" % v for v in np.abs(Zte[:, 60:]).max(0)])
acc = lambda Ztr, Zte_: KNeighborsClassifier(5).fit(Ztr, ytr).score(Zte_, yte)
ks = [2, 5, 10, 15, 20, 30, 40, 50, 55, 60, 61, 64]; plain, white, regw = [], [], []
for k in ks:
    q = PCA(k).fit(Xtr); a, b = q.transform(Xtr), q.transform(Xte); qw = PCA(k, whiten=True).fit(Xtr)
    plain.append(acc(a, b)); white.append(acc(qw.transform(Xtr), qw.transform(Xte))); regw.append(acc(a / np.sqrt(np.maximum(q.explained_variance_, 0) + 1.0), b / np.sqrt(np.maximum(q.explained_variance_, 0) + 1.0)))
print("  k  plain  whitened  whitened with sqrt(lambda + 1)")
for k, a, b, c in zip(ks, plain, white, regw): print("%3d  %.4f   %.4f   %.4f" % (k, a, b, c))
print("\nfixes at 64 components:")
for e in (0.1, 1.0, 10.0):
    q = PCA(64).fit(Xtr); s_ = np.sqrt(np.maximum(q.explained_variance_, 0) + e); print("  scale sqrt(lambda + %4.1f): %.4f" % (e, acc(q.transform(Xtr) / s_, q.transform(Xte) / s_)))
vt = VarianceThreshold().fit(Xtr); A, B = vt.transform(Xtr), vt.transform(Xte); q = PCA(60, whiten=True).fit(A); print("  drop the 4 constant pixels, PCA(60, whiten=True): %.4f" % acc(q.transform(A), q.transform(B)))
for f in (0.95, 0.99):
    q = PCA(f, whiten=True).fit(Xtr); print("  PCA(%.2f, whiten=True) keeps %d components: %.4f" % (f, q.n_components_, acc(q.transform(Xtr), q.transform(Xte))))
q = PCA(64).fit(Xtr); print("  no whitening, 64 components: %.4f" % acc(q.transform(Xtr), q.transform(Xte)))

fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.9), dpi=240)
a_ = axs[0]; j = np.arange(1, 65); sc = np.sqrt(np.maximum(lam, 1e-30)); sc = np.maximum(sc, eps)
a_.bar(j, 1 / sc, color=[BLUE if v > 1e-10 else RED for v in lam], width=0.8); a_.set_yscale('log'); a_.set_ylim(1e-2, 1e17); a_.set_yticks([1e-1, 1e3, 1e7, 1e11, 1e15])
a_.set_xlabel('component $j$'); a_.set_ylabel('weight  $1/\\sqrt{\\lambda_j}$'); a_.set_title('what whitening multiplies by', fontsize=8.6)
a_.annotate('four zero\neigenvalues', (61.5, 1e14), (33, 1e11), fontsize=8.0, color=RED, arrowprops=dict(arrowstyle='-', color=RED, lw=0.7), ha='center')
a_ = axs[1]; a_.plot(ks, plain, '-o', color=BLUE, ms=2.8, lw=1.1, label='no whitening'); a_.plot(ks, white, '-s', color=OR, ms=2.8, lw=1.1, label='whiten=True'); a_.plot(ks, regw, '-^', color=GREEN, ms=2.8, lw=1.1, label=r'$1/\sqrt{\lambda_j+1}$')
a_.set_xlabel('components $k$'); a_.set_ylabel('test accuracy'); a_.set_ylim(0, 1.05); a_.legend(fontsize=8.0, loc='lower left', borderaxespad=0.2, handlelength=1.4); a_.set_title('5-NN on the test images', fontsize=8.6)
for a_ in axs: a_.grid(color=GRID, lw=0.5); a_.set_axisbelow(True); a_.tick_params(labelsize=8.3); a_.xaxis.label.set_size(8.6); a_.yaxis.label.set_size(8.6)
fig.subplots_adjust(wspace=0.38)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s42c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s42c1.png') as im: w_ = im.size[0] / 240
print("\nfigure width %.2f in, printed 5.0 in, smallest font %.1f pt -> printed %.1f pt" % (w_, fs, fs * 5.0 / w_))
