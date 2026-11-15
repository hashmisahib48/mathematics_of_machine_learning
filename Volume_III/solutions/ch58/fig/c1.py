# Problem 58C.1: a forest's scores are not probabilities. Writes fig/s58c1.png, verify/c1.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)                    # 0 = malignant
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
mk = lambda: RF(n_estimators=200, random_state=0)
def cost(yt, p, t): yes = p >= t; return (10 * ((~yes) & (yt == 0)).sum() + 1 * (yes & (yt == 1)).sum()) / len(yt)

# --- test rows: reliability in five equal-width bins
forest = mk().fit(Xa, ya); p_rf = forest.predict_proba(Xb)[:, 0]
cal = CalibratedClassifierCV(mk(), method='sigmoid', cv=5).fit(Xa, ya); p_cal = cal.predict_proba(Xb)[:, 0]
p_lr = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xa, ya).predict_proba(Xb)[:, 0]
edges = np.linspace(0, 1, 6)
def rel(p):
    b = np.digitize(p, edges[1:-1]); out = []
    for k in range(5):
        m = b == k
        if m.sum(): out.append((p[m].mean(), (yb[m] == 0).mean(), m.sum()))
    return np.array(out)
R = {'forest': rel(p_rf), 'forest, sigmoid': rel(p_cal), 'logistic': rel(p_lr)}
for k, v in R.items(): print(k, "bins (mean score, observed share, n):", [(round(float(a), 3), round(float(b), 3), int(n)) for a, b, n in v])
# --- training rows: out-of-fold cost against the threshold (miss = 10 false alarms)
ts = np.r_[0.005, np.linspace(0.01, 0.6, 120)]
oof_rf = cross_val_predict(mk(), Xa, ya, cv=5, method='predict_proba')[:, 0]
oof_cal = cross_val_predict(CalibratedClassifierCV(mk(), method='sigmoid', cv=5), Xa, ya, cv=5, method='predict_proba')[:, 0]
C = {}
for nm, p in (('forest', oof_rf), ('forest, sigmoid', oof_cal)):
    c = np.array([cost(ya, p, t) for t in ts]); C[nm] = c; k = c.argmin()
    print("%-16s out-of-fold cost at t* = 1/11: %.4f ; minimum %.4f at t = %.3f ; cost at 0.5: %.4f" % (nm, cost(ya, p, 1 / 11), c.min(), ts[k], cost(ya, p, 0.5)))
print("out-of-fold share of forest scores below 0.05: %.3f, above 0.95: %.3f" % ((oof_rf < .05).mean(), (oof_rf > .95).mean()))
print("test: forest scores equal to 0:", (p_rf == 0).sum(), " forest minimum nonzero:", round(p_rf[p_rf > 0].min(), 4))

fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.95), dpi=240)
a = axs[0]; a.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls=':')
for (nm, v), col, mkr, ls in zip(R.items(), (PUR, GREEN, BLUE), ('o', '^', 's'), ('-', '--', '-.')):
    a.plot(v[:, 0], v[:, 1], ls=ls, color=col, lw=1.1); a.scatter(v[:, 0], v[:, 1], s=8 + 3.0 * np.sqrt(v[:, 2]), color=col, marker=mkr, zorder=3, label=nm)
a.set_xlabel('predicted probability'); a.set_ylabel('share malignant'); a.set_xticks([0, 0.5, 1]); a.set_yticks([0, 0.5, 1])
a.legend(loc='upper left', fontsize=8.2, handletextpad=0.1, borderaxespad=0.1, labelspacing=0.2, markerscale=0.6)
a = axs[1]
for (nm, c), col, ls in zip(C.items(), (PUR, GREEN), ('-', '--')): a.plot(ts, c, color=col, ls=ls, lw=1.3, label=nm)
a.axvline(1 / 11, color=RED, lw=0.9); a.text(1 / 11 + 0.015, 0.31, '$t^{*}$', fontsize=8.6, color=RED, va='top')
a.set_xlabel('threshold'); a.set_ylabel('cost per patient'); a.set_xlim(0, 0.6); a.set_ylim(0, 0.5); a.legend(loc='upper right', fontsize=8.2, handlelength=1.6, borderaxespad=0.1, labelspacing=0.2)
for a in axs: a.grid(color=GRID, lw=0.5); a.set_axisbelow(True); a.tick_params(labelsize=8.3); a.xaxis.label.set_size(8.6); a.yaxis.label.set_size(8.6)
fig.subplots_adjust(wspace=0.32)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s58c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s58c1.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))
