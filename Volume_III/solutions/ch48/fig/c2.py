# Problem 48C.2: the base rate. Writes fig/s48c2.png, verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); sys.path.insert(0, HERE + '/../../../ch48/code'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sensors import make_sensors

n, nf = 10000, 20                                     # one day: 0.2 per cent faulty
print("readings", n, " faults", nf, " share", nf / n)
print("contamination 0.05: flagged", int(0.05 * n), " best precision", nf / (0.05 * n), " false alarms at least", int(0.05 * n) - nf)
X, y = make_sensors(n=n - nf, n_fault=nf, seed=11)
Z = StandardScaler().fit_transform(X)
f = IsolationForest(random_state=0).fit(Z); s = -f.score_samples(Z)
print("AUC of the forest on the day:", round(roc_auc_score(y, s), 4), "  (a good detector)")
order = np.argsort(-s)
print("gamma     flagged  caught  precision  recall  upper bound min(1, 20/flagged)")
G = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]; prec = []; rec = []; ub = []
for g in G:
    k = int(round(g * n)); flag = np.zeros(n, bool); flag[order[:k]] = True
    tp = int((flag & (y == 1)).sum()); prec.append(tp / k); rec.append(tp / nf); ub.append(min(1, nf / k))
    m = IsolationForest(contamination=g, random_state=0).fit(Z); lab = (m.predict(Z) == -1)
    print(f"{g:6.3f} {k:9d} {tp:7d} {tp / k:10.3f} {tp / nf:7.2f} {min(1, nf / k):10.3f}    library 'contamination' flags {int(lab.sum())}, precision {float((lab & (y == 1)).sum() / lab.sum()):.3f}")
print("\nBayes: P(fault | flagged) = recall*0.002 / (recall*0.002 + FPR*0.998)")
k = 500; tp = 20; fpr = (k - tp) / (n - nf)
print("with recall 1 and a false-alarm rate of %.4f: %.4f" % (fpr, 20 / (20 + fpr * (n - nf))))
print("to reach a precision of 50 %% with 20 faults and recall 1 the flags must not exceed %d; the false-alarm rate must be at most %.5f" % (40, 20 / (n - nf)))
print("flags per day for a review capacity of 40: contamination", 40 / n)

fig, ax = plt.subplots(figsize=(3.4, 1.9), dpi=240)
ax.plot(G, prec, '-o', color=BLUE, ms=3.2, lw=1.2, label='precision'); ax.plot(G, rec, '-s', color=OR, ms=3.2, lw=1.2, label='recall')
ax.plot(G, ub, ls='--', color=GREY, lw=0.9, label='bound 20 / flagged'); ax.axvline(nf / n, color=GREEN, lw=0.9, ls=':')
ax.set_xscale('log'); ax.set_xticks([0.001, 0.002, 0.005, 0.01, 0.05, 0.1]); ax.set_xticklabels(['0.001', '0.002', '0.005', '0.01', '0.05', '0.1']); ax.minorticks_off()
ax.set_xlabel('contamination $\\gamma$ (dotted: the true share 0.002)', fontsize=8.2); ax.set_ylabel('share', fontsize=8.2); ax.set_ylim(0, 1.08); ax.tick_params(labelsize=7.8)
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); ax.legend(fontsize=7.8, loc='upper center', ncol=3, handlelength=1.5, bbox_to_anchor=(0.5, -0.30), borderaxespad=0.1, columnspacing=1.0)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s48c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s48c2.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 3.4 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 3.4 / w))
