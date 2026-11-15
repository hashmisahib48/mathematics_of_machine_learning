# Problem 48C.1: the reading between two modes. Writes fig/s48c1.png, verify/c1.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

rng = np.random.RandomState(0)                     # the data of the problem
P = np.vstack([rng.normal([-3, 0], 0.4, (150, 2)), rng.normal([3, 0], 0.4, (150, 2))]); mid = np.array([[0.0, 0.0]])
dets = [('isolation forest', IsolationForest(random_state=0)), ('LOF, $k$ = 20', LocalOutlierFactor(novelty=True)),
        ('one-class SVM, $\\gamma$ = 0.1', OneClassSVM(nu=0.05, gamma=0.1)), ('elliptic envelope', EllipticEnvelope(contamination=0.05, random_state=0))]
print("detector                 verdict  decision(0,0)  offset-free score(0,0)  share of training scores below the score of (0,0)")
for nm, d in dets:
    d.fit(P); s = d.score_samples(mid)[0]; tr = d.score_samples(P)
    print(f"{nm.replace('$',''):30s} {int(d.predict(mid)[0]):3d} {d.decision_function(mid)[0]:12.3f} {s:12.3f} {np.mean(tr < s):10.3f}")
ee = dets[3][1]
print("elliptic envelope: location", ee.location_.round(3), " covariance diag", np.diag(ee.covariance_).round(3), " d2 of (0,0):", round(float(ee.mahalanobis(mid)[0]), 3), " 5% threshold of d2:", round(float(-ee.offset_), 2))
print("pooled variance of x: ", round(float(P[:, 0].var()), 3), " (= 9 + 0.16 expected: %.2f)" % (9 + 0.16))
print("\none-class SVM, the verdict at (0,0) against gamma (nu = 0.05):")
for g in (0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 3.0):
    o = OneClassSVM(nu=0.05, gamma=g).fit(P)
    k = np.exp(-g * 9.0)
    print(f"  gamma {g:5.2f}  kernel value to a mode centre exp(-9 gamma) = {k:.4f}  verdict {int(o.predict(mid)[0]):3d}  decision {o.decision_function(mid)[0]:8.3f}  support vectors {len(o.support_)}")
print("\nLOF at (0,0) against k: neighbours on each side, distance to the k-th neighbour, median k-distance of the training points:")
for k in (20, 100, 149, 151, 299):
    l = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(P)
    nb = l.kneighbors(mid, n_neighbors=k, return_distance=False)[0]; dq = l.kneighbors(mid, n_neighbors=k)[0][0, -1]
    kd = LocalOutlierFactor(n_neighbors=k).fit(P)._distances_fit_X_[:, -1]
    print(f"  k = {k:3d}  left {int((P[nb, 0] < 0).sum()):3d} right {int((P[nb, 0] > 0).sum()):3d}  d_k(0,0) = {dq:5.2f}  median d_k of the training points = {np.median(kd):5.2f}  -LOF(0,0) = {l.score_samples(mid)[0]:8.3f}  verdict {int(l.predict(mid)[0]):3d}")
print("\nisolation forest: path length of (0,0) against a training point, 1000 trees, and the share of random cuts that separate the modes")
f = IsolationForest(n_estimators=1000, random_state=0).fit(P)
print("  mean s of the training points %.3f, s of (0,0) %.3f, the 95%% quantile of training s %.3f" % (-f.score_samples(P).mean(), -f.score_samples(mid)[0], np.quantile(-f.score_samples(P), 0.95)))
sub = 256; w = (P[:, 0].max() - P[:, 0].min()); gap_share = (np.sort(np.abs(P[:, 0]))[0] * 2) / w
print("  range of x %.2f; the empty gap between the modes (2 x smallest |x|) is %.2f, i.e. %.0f%% of the range" % (w, np.sort(np.abs(P[:, 0]))[0] * 2, 100 * gap_share))

xx, yy = np.meshgrid(np.linspace(-5, 5, 140), np.linspace(-2.5, 2.5, 70)); G = np.c_[xx.ravel(), yy.ravel()]
fig, axs = plt.subplots(1, 4, figsize=(5.0, 1.45), dpi=240, sharey=True)
for a, (nm, d) in zip(axs, dets):
    dfn = d.decision_function(G).reshape(xx.shape)
    a.contour(xx, yy, dfn, levels=[0], colors=[GREEN], linewidths=1.2)
    a.scatter(P[:, 0], P[:, 1], s=1.5, color=BLUE, lw=0)
    ok = d.predict(mid)[0] == 1
    a.plot(0, 0, 'o' if ok else 'x', color=RED, ms=5 if ok else 6, mew=1.6, mfc='none' if ok else RED)
    a.set_title(nm + ('\n(0,0) normal' if ok else '\n(0,0) strange'), fontsize=7.6, pad=2)
    a.set_xlim(-5, 5); a.set_ylim(-2.5, 2.5); a.set_xticks([-4, 0, 4]); a.set_yticks([-2, 0, 2]); a.tick_params(labelsize=7.6)
axs[0].set_ylabel('feature 2', fontsize=8); fig.supxlabel('feature 1', fontsize=8, y=-0.13)
fig.subplots_adjust(wspace=0.12, top=0.80)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s48c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s48c1.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))
