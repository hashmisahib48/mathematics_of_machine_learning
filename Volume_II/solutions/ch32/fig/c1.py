# Problem 32C.1: calibrated but useless. Writes fig/s32c1.png, verify/c1.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

rng = np.random.default_rng(0)                     # the data of Section 32.4
Z = rng.normal(size=(25000, 3)); X = np.hstack([Z] + [Z + 0.3 * rng.normal(size=Z.shape) for _ in range(3)])
y = (rng.random(25000) < 1 / (1 + np.exp(-Z @ [1.6, 1.28, 0.96]))).astype(int)
Xtr, ytr, Xte, yte = X[:5000], y[:5000], X[15000:], y[15000:]
const = np.full(len(yte), yte.mean()); lr = LogisticRegression().fit(Xtr, ytr).predict_proba(Xte)[:, 1]

def decomp(p, yv, bins=10):                         # Brier decomposition on bins of equal width
    k = np.minimum((p * bins).astype(int), bins - 1); o = yv.mean(); rel = res = 0
    for b in range(bins):
        m = k == b
        if m.any(): rel += m.sum() * (p[m].mean() - yv[m].mean()) ** 2; res += m.sum() * (yv[m].mean() - o) ** 2
    return rel / len(yv), res / len(yv), o * (1 - o)
print(f"{'forecaster':22s}   REL     RES     UNC   REL-RES+UNC  Brier  log loss   AUC")
for name, p in (("constant (rate %.4f)" % yte.mean(), const), ("logistic regression", lr)):
    rl, rs, u = decomp(p, yte)
    print(f"{name:22s} {rl:7.4f} {rs:7.4f} {u:7.4f} {rl - rs + u:9.4f} {brier_score_loss(yte, p):8.4f} {log_loss(yte, p):8.4f} {roc_auc_score(yte, p):7.4f}")
print("ln 2 =", round(float(np.log(2)), 4), " ; area under the ROC curve of any constant: 0.5")
print("a constant forecaster with the wrong value 0.9: Brier %.4f, log loss %.4f" % (brier_score_loss(yte, np.full(len(yte), .9)), log_loss(yte, np.full(len(yte), .9))))

fig, axs = plt.subplots(1, 2, figsize=(4.6, 1.95), dpi=240)
a = axs[0]; a.plot([0, 1], [0, 1], color=GREY, ls='--', lw=0.8)
t, q = calibration_curve(yte, lr, n_bins=10, strategy='quantile')
a.plot(q, t, '-o', color=GREEN, ms=3, lw=1.1, label='logistic regression'); a.plot([yte.mean()], [yte.mean()], 's', color=RED, ms=6, label='constant forecast')
a.set_xlabel('forecast'); a.set_ylabel('observed frequency'); a.legend(fontsize=7.8, loc='upper left'); a.set_xlim(0, 1); a.set_ylim(0, 1)
a = axs[1]
for p, col, lab in ((lr, GREEN, 'logistic regression'), (const, RED, 'constant forecast')):
    fp, tp, _ = roc_curve(yte, p); a.plot(fp, tp, color=col, lw=1.3, label=lab)
a.plot([0, 1], [0, 1], color=GREY, ls='--', lw=0.8); a.set_xlabel('false positive rate'); a.set_ylabel('true positive rate'); a.legend(fontsize=7.8, loc='lower right')
for a in axs: a.grid(color=GRID, lw=0.5); a.set_axisbelow(True); a.tick_params(labelsize=7.8); a.xaxis.label.set_size(8.2); a.yaxis.label.set_size(8.2)
fig.subplots_adjust(wspace=0.38)
fs = min(t_.get_fontsize() for t_ in fig.findobj(mtext.Text) if t_.get_text().strip() and t_.get_visible())
fig.savefig('fig/s32c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s32c1.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.6 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.6 / w))
