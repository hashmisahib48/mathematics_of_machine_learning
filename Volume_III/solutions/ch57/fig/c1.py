# Problem 57C.1: AUC 0.99 and recall 0.5. Writes fig/s57c1.png and verify/c1.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

d = 3.3
print("AUC of two normal scores d = %.1f apart: Phi(d/sqrt2) = %.4f" % (d, norm.cdf(d / np.sqrt(2))))
tpr = lambda t: norm.sf(t - d); fpr = lambda t: norm.sf(t)
print("a threshold at the median of the positives, t = d: recall %.3f  FPR %.5f" % (tpr(d), fpr(d)))
pi = 1 / (1 + np.exp(d * d / 2))
print("posterior probability of a positive is 0.5 at t = d when prevalence is pi = 1/(1+exp(d^2/2)) = %.5f" % pi)
prec = lambda t, p=pi: p * tpr(t) / (p * tpr(t) + (1 - p) * fpr(t))
print("precision at t = d: %.3f" % prec(d))
# a miss costs 20 false alarms: flag when posterior >= 1/21, i.e. LR >= (1/20) (1-pi)/pi, LR = exp(d t - d^2/2)
t_cost = (d * d / 2 + np.log((1 - pi) / pi / 20)) / d
print("cost rule (miss = 20 false alarms): posterior >= 1/21 -> t = %.3f  recall %.3f  FPR %.5f  precision %.3f"
      % (t_cost, tpr(t_cost), fpr(t_cost), prec(t_cost)))
n_ill_per_1e5 = pi * 1e5
print("per 100000 people: %.0f ill; default flags %.0f ill + %.0f well; cost rule flags %.0f ill + %.0f well"
      % (n_ill_per_1e5, n_ill_per_1e5 * tpr(d), (1e5 - n_ill_per_1e5) * fpr(d), n_ill_per_1e5 * tpr(t_cost), (1e5 - n_ill_per_1e5) * fpr(t_cost)))
# simulation: logistic regression on the one score, default predict()
rng = np.random.RandomState(0); n = 2_000_000
y = (rng.uniform(size=n) < pi).astype(int); s = rng.normal(size=n) + d * y
lr = LogisticRegression().fit(s[:, None], y); yh = lr.predict(s[:, None])
print("simulation, n = %d, positives %d: AUC %.4f  default predict(): recall %.3f  precision %.3f ; coef %.3f (theory %.3f)"
      % (n, y.sum(), roc_auc_score(y, s), yh[y == 1].mean(), y[yh == 1].mean(), lr.coef_[0, 0], d))
pm = lr.predict_proba(s[:, None])[:, 1]; yc = pm >= 1 / 21
print("cost rule on the fitted probabilities (>= 1/21): recall %.3f  false alarms per 100000 well %.0f  precision %.3f"
      % (yc[y == 1].mean(), 1e5 * yc[y == 0].mean(), y[yc].mean()))

fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.95), dpi=240)
t = np.linspace(-3, 8, 800); a = axs[0]
a.plot(fpr(t), tpr(t), color=GREEN, lw=1.4); a.set_xscale('log'); a.set_xlim(1e-4, 1); a.set_ylim(0, 1.03)
a.plot(fpr(d), tpr(d), 'o', color=RED, ms=4.5); a.annotate('default\nrecall 0.50', (fpr(d), tpr(d)), (2e-4, 0.17), fontsize=8.2, color=RED, arrowprops=dict(arrowstyle='-', color=RED, lw=0.6))
a.plot(fpr(t_cost), tpr(t_cost), 's', color=BLUE, ms=4.5); a.text(2.5e-2, 0.52, 'cost rule\nrecall %.2f' % tpr(t_cost), fontsize=8.2, color=BLUE)
a.set_xlabel('FPR (logarithmic)'); a.set_ylabel('TPR')
a = axs[1]; t2 = np.linspace(0, 6, 400)
a.plot(t2, tpr(t2), color=GREEN, lw=1.4, label='recall'); a.plot(t2, prec(t2), color=PUR, lw=1.4, ls='--', label='precision')
a.axvline(d, color=RED, lw=0.9); a.axvline(t_cost, color=BLUE, lw=0.9, ls='-.')
a.set_xlabel('threshold t on the score'); a.set_ylim(0, 1.03)
a.text(4.15, 0.42, 'recall', color=GREEN, fontsize=8.2); a.text(4.05, 0.84, 'precision', color=PUR, fontsize=8.2)
for a in axs: a.grid(color=GRID, lw=0.5); a.set_axisbelow(True); a.tick_params(labelsize=8.3); a.xaxis.label.set_size(8.6); a.yaxis.label.set_size(8.6)
fig.subplots_adjust(wspace=0.3)
fs = min(t_.get_fontsize() for t_ in fig.findobj(mtext.Text) if t_.get_text().strip() and t_.get_visible())
fig.savefig('fig/s57c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s57c1.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))
