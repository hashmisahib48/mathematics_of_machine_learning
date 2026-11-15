# Problem 57C.2: which scores of a balanced validation carry over to 5 % prevalence. Writes fig/s57c2.png, verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); sys.path.insert(0, HERE + '/../../../ch57/code'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from scipy.stats import norm
from sklearn import metrics as M
from sklearn.linear_model import LogisticRegression
from screendata import make_screen

d, t = 2.0, 1.5
tpr, fpr = norm.sf(t - d), norm.sf(t); spec = 1 - fpr
auc = norm.cdf(d / np.sqrt(2))
def counts(pi, n=1.0):
    TP, FN, FP, TN = pi * tpr, pi * (1 - tpr), (1 - pi) * fpr, (1 - pi) * spec
    return TP, FN, FP, TN
rows = {}
for pi in (0.5, 0.05):
    TP, FN, FP, TN = counts(pi)
    prec = TP / (TP + FP); acc = TP + TN; f1 = 2 * TP / (2 * TP + FN + FP)
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    s, y = make_screen(400000, pi, d, 1); ap = M.average_precision_score(y, s); a_ = M.roc_auc_score(y, s)
    rows[pi] = dict(recall=tpr, specificity=spec, balanced=(tpr + spec) / 2, AUC=auc, precision=prec, accuracy=acc, F1=f1, MCC=mcc, AP=ap)
    print("pi = %.2f:" % pi, "  ".join("%s %.3f" % (k, v) for k, v in rows[pi].items()), " (simulated AUC %.3f)" % a_)
print("LR+ = %.2f is the same in both populations" % (tpr / fpr))
# probabilities: the prior odds change by a factor
f = (0.05 / 0.95) / (0.5 / 0.5); print("odds factor (0.05/0.95)/(0.5/0.5) = %.4f" % f)
for p in (0.5, 0.8, 0.95):
    o = p / (1 - p) * f; print("  validation probability %.2f -> deployment probability %.4f" % (p, o / (1 + o)))
# best thresholds for F1 in the two populations (exact expected counts on a grid)
ts = np.linspace(-1, 5, 6001)
for pi in (0.5, 0.05):
    TPr, FPr = norm.sf(ts - d), norm.sf(ts)
    TPc, FNc, FPc = pi * TPr, pi * (1 - TPr), (1 - pi) * FPr
    F1 = 2 * TPc / (2 * TPc + FNc + FPc); k = np.argmax(F1)
    print("pi %.2f: F1 is largest at t = %.2f (F1 %.3f, recall %.3f, precision %.3f)" % (pi, ts[k], F1[k], TPr[k], TPc[k] / (TPc[k] + FPc[k])))
# a logistic regression fitted on the balanced sample, probabilities used at 5 % prevalence
sb, yb = make_screen(100000, 0.5, d, 5); sd, yd = make_screen(100000, 0.05, d, 6)
lr = LogisticRegression().fit(sb[:, None], yb)
pb = lr.predict_proba(sb[:, None])[:, 1]; pd_ = lr.predict_proba(sd[:, None])[:, 1]
print("balanced-fit probabilities: log loss on balanced data %.4f, on 5 %% data %.4f; mean forecast %.3f against prevalence %.3f"
      % (M.log_loss(yb, pb), M.log_loss(yd, pd_), pd_.mean(), yd.mean()))
od = pd_ / (1 - pd_) * f; pc = od / (1 + od)
print("log loss of the constant 5 %% forecast: %.4f" % M.log_loss(yd, np.full(len(yd), yd.mean())))
print("after multiplying the odds by %.4f: log loss %.4f, mean forecast %.3f" % (f, M.log_loss(yd, pc), pc.mean()))

keep = ['recall', 'specificity', 'AUC', 'precision', 'F1', 'MCC', 'AP', 'accuracy']
lab = ['recall', 'specificity', 'AUC', 'precision', 'F1', 'MCC', 'AP', 'accuracy']
fig, ax = plt.subplots(figsize=(4.9, 1.9), dpi=240); x = np.arange(len(keep)); w = 0.38
ax.bar(x - w / 2, [rows[0.5][k] for k in keep], w, color=BLUE, label='validation, 50 % ill')
ax.bar(x + w / 2, [rows[0.05][k] for k in keep], w, color=OR, label='deployment, 5 % ill')
ax.axvline(2.5, color=GREY, lw=0.8, ls=':'); ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8.2); ax.set_ylim(0, 1.18); ax.set_yticks([0, 0.5, 1])
ax.tick_params(labelsize=8.2); ax.legend(loc='upper right', ncol=2, fontsize=8.2, handlelength=1.0, columnspacing=1.0, borderaxespad=0.1)
ax.text(1.0, 1.04, 'unchanged', ha='center', fontsize=8.2, color=NAVY); ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(pad=0.4)
fs = min(t_.get_fontsize() for t_ in fig.findobj(mtext.Text) if t_.get_text().strip() and t_.get_visible())
fig.savefig('fig/s57c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s57c2.png') as im: w_ = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w_, fs, fs * 4.9 / w_))
