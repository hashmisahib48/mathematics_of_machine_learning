# Problem figures of Chapter 32. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys, os, json, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../lib'); os.chdir(HERE + '/..')
warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
from scipy.special import softmax
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.isotonic import IsotonicRegression, isotonic_regression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
plt.rcParams.update({'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'axes.labelsize': 10, 'legend.fontsize': 9.5})
A = {}; CHK = {}; WPX = 110
sg = lambda z: 1 / (1 + np.exp(-z))
def small(w=2.0, h=1.35): return plt.subplots(figsize=(w, h), dpi=240)
def sv(fig, name):
    import matplotlib.text as mtext
    fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()); save(fig, name)
    from PIL import Image; w, h = Image.open(name).size; pw = min(1.55 * WPX, 235 * w / h); CHK[name] = round(fs * pw / 96 / (w / 240), 1)

# 32A.1 weighted PAVA
y = np.array([6, 2, 4, 3, 8.0]); w = np.array([1, 1, 2, 1, 1.0]); f = isotonic_regression(y, sample_weight=w)
A['32A.1'] = {'fit': f.round(4).tolist(), 'sum_wy': float((w * y).sum()), 'sum_wf': float((w * f).sum())}
fig, ax = small(); xs = np.arange(1, 6)
ax.scatter(xs, y, s=18 * w * 2.2, color=BLUE, edgecolor='white', zorder=4); ax.plot([0.7, 4.3], [f[0]] * 2, color=GREEN, lw=2); ax.plot([4.7, 5.3], [f[4]] * 2, color=GREEN, lw=2)
ax.set_xlabel('$i$ (marker area $\\propto w_i$)'); ax.set_ylabel('$y_i$'); ax.set_xlim(0.4, 5.6); ax.set_ylim(0, 9.5); ax.set_xticks(xs); sv(fig, 'p_32A1.png')

# 32A.2 reliability and Brier
lv = np.array([0.8, 0.2]); nk = np.array([4, 6]); ht = np.array([3, 2]); N = nk.sum(); ok = ht / nk; o = ht.sum() / N
rel = (nk * (lv - ok) ** 2).sum() / N; res = (nk * (ok - o) ** 2).sum() / N; unc = o * (1 - o); bs = rel - res + unc
yy = np.concatenate([[1] * h + [0] * (m - h) for h, m in zip(ht, nk)]); qq = np.repeat(lv, nk)
A['32A.2'] = {'obs': ok.round(4).tolist(), 'REL': round(float(rel), 4), 'RES': round(float(res), 4), 'UNC': round(float(unc), 4), 'BS': round(float(bs), 4), 'BS_direct': round(float(np.mean((yy - qq) ** 2)), 4)}
fig, ax = small(); ax.plot([0, 1], [0, 1], color=GRID, ls='--', lw=1.2); ax.plot(lv, ok, 'o-', color=BLUE, ms=5, lw=1.4)
for a_, b_ in zip(lv, ok): ax.text(a_ + 0.03, b_ - 0.02 if a_ < 0.5 else b_ - 0.09, f'{b_:.2f}', fontsize=9.5, color=NAVY, ha='left')
ax.set_xlabel('forecast $q$'); ax.set_ylabel('observed'); ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05); ax.set_xticks([0, 0.2, 0.5, 0.8, 1]); sv(fig, 'p_32A2.png')

# 32A.3 two Newton steps of Platt scaling
f3 = np.array([-2, -1, 1, 2.0]); l3 = np.array([0, 1, 0, 1]); t3 = np.where(l3 == 1, 3 / 4, 1 / 4); s = 0.0; steps = []
for k in range(4):
    p = sg(s * f3); g = np.sum(f3 * (p - t3)); h = np.sum(f3 ** 2 * p * (1 - p)); steps.append((round(s, 5), round(g, 5), round(h, 4))); s -= g / h
A['32A.3'] = {'steps(s,g,h)': steps, 's_final': round(float(s), 5), 'A': round(float(-s), 5), 'p_at_f1': round(float(sg(s * 1.0)), 4)}
fig, ax = small(); xx = np.linspace(-3, 3, 100); ax.plot(xx, sg(s * xx), color=GREEN, lw=1.8)
ax.scatter(f3, l3, s=24, color=[BLUE if v == 0 else OR for v in l3], marker='o', edgecolor='white', zorder=4); ax.scatter(f3, t3, s=22, marker='x', color=RED, zorder=4, lw=1.2)
ax.set_xlabel('score $f$'); ax.set_ylabel('$P(y=1)$'); ax.set_ylim(-0.05, 1.05); ax.set_xlim(-3, 3); sv(fig, 'p_32A3.png')

# 32A.4 temperature on three classes
z = np.array([2.0, 0.0, -1.0]); true = 1
res4 = {}
for b in (1.0, 0.5):
    P = softmax(b * z); res4[str(b)] = {'P': P.round(4).tolist(), 'loss': round(float(-np.log(P[true])), 4)}
A['32A.4'] = res4
fig, ax = small(); xk = np.arange(3); wd = 0.36
for j, (b, col, hat) in enumerate(((1.0, BLUE, ''), (0.5, GREEN, '//'))):
    ax.bar(xk + (j - 0.5) * wd, softmax(b * z), width=wd, color=col, hatch=hat, edgecolor='white', label=f'$\\beta$ = {b:g}')
ax.set_xticks(xk); ax.set_xticklabels(['1', '2 (true)', '3']); ax.set_ylabel('$P_k$'); ax.set_xlabel('class'); ax.set_ylim(0, 1.0); ax.legend(loc='upper right', fontsize=9.5); sv(fig, 'p_32A4.png')

# 32B.1 IsotonicRegression: auto direction and out_of_bounds
x = np.arange(1, 11.0); yd = np.array([9, 7, 8, 6, 6.5, 3, 4, 2, 1, 2.0])
m = IsotonicRegression(increasing='auto').fit(x, yd); mc = IsotonicRegression(increasing='auto', out_of_bounds='clip').fit(x, yd)
qx = [0, 3.5, 7.5, 11]
A['32B.1'] = {'increasing_': bool(m.increasing_), 'X_thresholds_': m.X_thresholds_.tolist(), 'y_thresholds_': m.y_thresholds_.round(4).tolist(), 'predict_nan': np.round(m.predict(qx), 4).tolist(), 'predict_clip': np.round(mc.predict(qx), 4).tolist()}
fig, ax = small(2.0, 1.1); ax.plot(m.X_thresholds_, m.y_thresholds_, color=GREEN, lw=2); ax.plot(x, yd, 'o', color=BLUE, ms=4.5, mec='white'); ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); sv(fig, 'p_32B1.png')

# 32B.2 calibration_curve: uniform against quantile bins on a rare class
from sklearn.linear_model import LogisticRegression
Xc, yc = make_classification(n_samples=4000, n_features=10, n_informative=3, weights=[0.9, 0.1], flip_y=0.02, random_state=0)
Xa, Xb, ya, yb = train_test_split(Xc, yc, test_size=0.5, random_state=0, stratify=yc)
pb = LogisticRegression().fit(Xa, ya).predict_proba(Xb)[:, 1]; out = {'positive_share': float(yb.mean())}
for st in ('uniform', 'quantile'):
    t, q = calibration_curve(yb, pb, n_bins=5, strategy=st)
    edges = np.linspace(0, 1, 6) if st == 'uniform' else np.percentile(pb, np.linspace(0, 100, 6))
    cnt = np.bincount(np.searchsorted(edges[1:-1], pb), minlength=5)
    out[st] = {'pred': q.round(3).tolist(), 'obs': t.round(3).tolist(), 'n_bins_returned': len(q), 'counts': cnt.tolist()}
A['32B.2'] = out
fig, ax = small(2.0, 1.1); ax.plot([0, 0.65], [0, 0.65], color=GRID, ls='--', lw=1.2)
for st, col, mk in (('uniform', BLUE, 'o'), ('quantile', OR, 's')): ax.plot(out[st]['pred'], out[st]['obs'], marker=mk, color=col, ms=4.5, lw=1.3, label=st)
ax.set_xlabel('predicted'); ax.set_ylabel('observed'); ax.legend(loc='upper left', fontsize=9.5); ax.set_xlim(-0.02, 0.66); ax.set_ylim(-0.03, 1.05); sv(fig, 'p_32B2.png')

# 32B.3 which predictions change?
X, yb3 = load_breast_cancer(return_X_y=True); Xt, Xe, yt, ye = train_test_split(X, yb3, test_size=0.25, random_state=0, stratify=yb3)
plain = make_pipeline(StandardScaler(), SVC()).fit(Xt, yt); base = plain.predict(Xe); ch = {}
for meth in ('sigmoid', 'isotonic', 'temperature'):
    c = make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(), method=meth, cv=5, ensemble=False)).fit(Xt, yt)
    ch[meth] = int((c.predict(Xe) != base).sum())
A['32B.3'] = {'changed': ch, 'svc_acc': float(plain.score(Xe, ye))}
fig, ax = small(2.0, 1.05); ax.bar(range(3), list(ch.values()), color=[GREEN, OR, PUR], width=0.6)
for k, v in enumerate(ch.values()): ax.text(k, v + 0.15, str(v), ha='center', fontsize=9.5, color=NAVY)
ax.set_xticks(range(3)); ax.set_xticklabels(['sigmoid', 'isotonic', 'temp.'], fontsize=9.5); ax.set_ylabel('changed classes'); ax.set_ylim(0, max(ch.values()) + 1.6); sv(fig, 'p_32B3.png')

# 32C.1 calibrated but useless
fig, ax = small(2.0, 1.95); ax.plot([0, 1], [0, 1], color=GRID, ls='--', lw=1.2); ax.plot(0.5, 0.5, 'o', color=BLUE, ms=7, zorder=4)
ax.text(0.55, 0.42, 'every forecast\nis 0.5', fontsize=9.5, color=NAVY, va='top'); ax.set_xlabel('forecast'); ax.set_ylabel('observed'); ax.set_xlim(0, 1); ax.set_ylim(0, 1); sv(fig, 'p_32C1.png')
# 32C.2 isotonic gives zero
fig, ax = small(2.0, 1.95); gx = np.array([-3, -1.5, -1, 1, 1.5, 3]); gy = np.array([0, 0, 0.5, 0.5, 1, 1.0]); ax.plot(gx, gy, color=OR, lw=2); ax.plot([-2.2], [1], 'o', color=RED, ms=6, zorder=5)
ax.annotate('an event\nwith $p=0$', xy=(-2.2, 1), xytext=(-1.6, 0.76), fontsize=9.5, color=RED, arrowprops=dict(arrowstyle='-|>', color=RED, lw=0.9), va='center', ha='left')
ax.set_xlabel('score $f$'); ax.set_ylabel('calibrated $p$'); ax.set_ylim(-0.05, 1.1); sv(fig, 'p_32C2.png')
# 32C.3 which data to calibrate on
fig, ax = small(2.0, 1.15); parts = [(0, 5000, BLUE, 'train'), (5000, 10000, GREEN, 'calibrate'), (15000, 10000, GREY, 'judge')]
for a_, w_, c_, t_ in parts:
    ax.barh(0, w_, left=a_, color=c_, edgecolor='white', height=0.45, alpha=0.9)
    ax.text(a_ if t_ == 'train' else a_ + w_ / 2, 0.4, t_, ha='left' if t_ == 'train' else 'center', va='bottom', fontsize=9.5, color=NAVY)
ax.set_xlim(0, 25000); ax.set_ylim(-0.4, 0.9); ax.set_yticks([]); ax.set_xticks([0, 5000, 15000, 25000]); ax.set_xticklabels(['0', '5 000', '15 000', '25 000'], fontsize=9.5); ax.set_xlabel('samples'); ax.spines['left'].set_visible(False); sv(fig, 'p_32C3.png')
json.dump(A, open('verify/problem_answers.json', 'w'), indent=1); json.dump(CHK, open('verify/probfig_textsize.json', 'w'), indent=1); print(json.dumps(A)); print(CHK)
