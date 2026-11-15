# All figures of Chapter 32. Run: python3 fig/figs.py  (PNGs into ch32/, numbers into verify/fig_numbers.json)
import sys, os, json, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../lib'); os.chdir(HERE + '/..')
warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.isotonic import IsotonicRegression, isotonic_regression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import log_loss, brier_score_loss
from PIL import Image
plt.rcParams.update({'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'axes.labelsize': 9, 'legend.fontsize': 8.5})
NUM = {}; CHK = {}
def sv(fig, name, printed_px):
    fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig, name); w = Image.open(name).size[0] / 240
    CHK[name] = {'fig_in': round(w, 2), 'printed_in': round(printed_px / 96, 2), 'min_font_design': round(fs, 1), 'min_font_printed': round(fs * printed_px / 96 / w, 1)}
sg = lambda z: 1 / (1 + np.exp(-z))
# ---- data of the book ---------------------------------------------------
dose = np.arange(1, 9.0); resp = np.array([2, 5, 3, 4, 1, 6, 5, 7.0])
def synth():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(25000, 3))
    X = np.hstack([Z] + [Z + 0.3 * rng.normal(size=Z.shape) for _ in range(3)])
    p_true = 1 / (1 + np.exp(-Z @ [1.6, 1.28, 0.96]))
    y = (rng.random(25000) < p_true).astype(int)
    return X, y
Xs, ys = synth()
X_tr, y_tr, X_ca, y_ca, X_te, y_te = Xs[:5000], ys[:5000], Xs[5000:15000], ys[5000:15000], Xs[15000:], ys[15000:]
nb = GaussianNB().fit(X_tr, y_tr); ada = AdaBoostClassifier(random_state=0).fit(X_tr, y_tr)
svm = LinearSVC().fit(X_tr, y_tr); lr = LogisticRegression().fit(X_tr, y_tr)
def curve(y, p, strategy='uniform', n=10): 
    t, q = calibration_curve(y, p, n_bins=n, strategy=strategy); return q, t
def hist_ax(ax, p, color=LGREY, h=0.18):
    """small histogram of predictions along the bottom of a reliability panel"""
    ax2 = ax.inset_axes([0, 0, 1, 0.15]); ax2.hist(p, bins=np.linspace(0, 1, 21), color='#B9C2CF', edgecolor='white', lw=0.3)
    ax2.set_xlim(0, 1); ax2.axis('off'); return ax2

# ---- 32.0-1 opening figure: isotonic fit and a reliability diagram ----------
fig, axs = plt.subplots(1, 2, figsize=(4.9, 2.35), dpi=240)
ax = axs[0]
iso = IsotonicRegression().fit(dose, resp); fitv = isotonic_regression(resp)
ax.plot(iso.X_thresholds_, iso.y_thresholds_, color=GREEN, lw=2.2, zorder=3)
for x_, y_, f_ in zip(dose, resp, fitv): ax.plot([x_, x_], [y_, f_], color=RED, lw=0.9, alpha=0.7, zorder=2)
ax.scatter(dose, resp, s=26, color=BLUE, edgecolor='white', lw=0.7, zorder=5)
ax.text(1.05, 6.9, 'fit never falls', color=GREEN, fontsize=8.5, weight='bold')
ax.set_xlabel('dose'); ax.set_ylabel('response'); ax.set_xlim(0.6, 8.4); ax.set_ylim(0.5, 7.8); ax.set_title('isotonic regression', fontsize=9.5, color=NAVY)
ax = axs[1]
cal_t = CalibratedClassifierCV(FrozenEstimator(nb), method='temperature').fit(X_ca, y_ca)
p_raw = nb.predict_proba(X_te)[:, 1]; p_cal = cal_t.predict_proba(X_te)[:, 1]
ax.plot([0, 1], [0, 1], color=GRID, lw=1.2, ls='--', zorder=1)
for p_, col, mk, ls, lab in ((p_raw, BLUE, 'o', '-', 'naive Bayes'), (p_cal, GREEN, 's', '-', 'after calibration')):
    q, t = curve(y_te, p_, 'quantile', 8); ax.plot(q, t, ls=ls, marker=mk, color=col, ms=4, lw=1.5, label=lab, zorder=4)
ax.legend(loc='lower right', fontsize=8.3, handlelength=1.6, borderaxespad=0.2)
ax.set_xlabel('predicted probability'); ax.set_ylabel('observed frequency'); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_title('reliability diagram', fontsize=9.5, color=NAVY)
ax.set_aspect('equal'); fig.tight_layout(w_pad=1.0); sv(fig, 'f320.png', 470)
NUM['opener'] = {'raw_logloss': float(log_loss(y_te, p_raw)), 'cal_logloss': float(log_loss(y_te, p_cal))}

# ---- 32.2-1 PAVA in four snapshots -------------------------------------------
snap = [([2, 5, 3, 4, 1, 6, 5, 7.0], 'data: three pairs break the order', None),
        ([2, 4, 4, 4, 1, 6, 5, 7.0], 'dose 3: 5 > 3, pool to 4', (2, 3)),
        ([2, 3.25, 3.25, 3.25, 3.25, 6, 5, 7.0], 'dose 5: 1 pulls the left blocks', (2, 5)),
        (list(isotonic_regression(resp)), 'dose 7: 6 > 5, pool to 5.5', (6, 7))]
fig, axs = plt.subplots(2, 2, figsize=(4.9, 3.6), dpi=240, sharex=True, sharey=True)
for k, (ax, (fv, ttl, hl)) in enumerate(zip(axs.ravel(), snap)):
    fv = np.array(fv)
    if k == 0:
        for a_, b_ in ((2, 3), (4, 5), (6, 7)): ax.axvspan(a_ - 0.4, b_ + 0.4, color=PINK, zorder=0)
    else:
        ax.axvspan(hl[0] - 0.4, hl[1] + 0.4, color=LB, zorder=0)
        i = 0
        while i < 8:
            j = i
            while j + 1 < 8 and fv[j + 1] == fv[i]: j += 1
            ax.plot([i + 0.7, j + 1.3], [fv[i], fv[i]], color=GREEN, lw=2.2, zorder=3)
            if j > i: ax.text((i + j) / 2 + 1, fv[i] + 0.35, f'{fv[i]:g}', ha='center', fontsize=8.3, color=GREEN, weight='bold')
            i = j + 1
    for x_, y_ in zip(dose, resp): ax.plot(x_, y_, 'o', color=BLUE, ms=4.2, mec='white', mew=0.6, zorder=5)
    ax.set_title(ttl, fontsize=8.6, color=NAVY); ax.set_xlim(0.4, 8.6); ax.set_ylim(0, 8)
for ax in axs[1]: ax.set_xlabel('dose')
for ax in axs[:, 0]: ax.set_ylabel('response')
fig.tight_layout(h_pad=0.8, w_pad=0.6); sv(fig, 'f322.png', 470)

# ---- 32.3-1 projection onto the cone f1 <= f2 -------------------------------
fig, ax = plt.subplots(figsize=(3.1, 2.9), dpi=240)
g = np.array([-0.5, 7.5]); ax.fill_between(g, g, 7.5, color=LB, zorder=0)
ax.plot(g, g, color=NAVY, lw=1.2); ax.text(6.6, 6.0, 'boundary\n$f_1=f_2$', color=NAVY, fontsize=8.5, ha='center', va='top', rotation=0)
ax.text(1.0, 6.5, 'allowed:\n$f_1\\leq f_2$', color=BLUE, fontsize=8.8, ha='center', va='center')
yv = np.array([5.0, 3.0]); pv = np.array([4.0, 4.0]); ax.add_patch(plt.Circle(tuple(yv), np.sqrt(2), fill=False, ec=RED, lw=1.1, ls=':', zorder=2))
ax.plot(*yv, 'o', color=BLUE, ms=6, mec='white', zorder=5); arr(ax, tuple(yv), tuple(pv + 0.12 * (yv - pv)), c=RED, lw=1.3, ms=8); ax.plot(*pv, 's', color=GREEN, ms=7, mec='white', zorder=6)
ax.text(5.0, 1.45, '$\\mathbf{y}=(5,3)$', color=BLUE, fontsize=9, ha='center', va='top'); ax.text(3.75, 4.35, '(4, 4)', color=GREEN, fontsize=9, ha='right', va='bottom', weight='bold')
yq = np.array([2.0, 5.0]); ax.plot(*yq, 'o', color=BLUE, ms=6, mec='white', zorder=5); ax.text(2.0, 5.3, '$(2,5)$ stays', color=BLUE, fontsize=8.5, ha='center', va='bottom')
ax.set_xlim(-0.5, 7.5); ax.set_ylim(-0.5, 7.5); ax.set_aspect('equal'); ax.set_xlabel('$f_1$'); ax.set_ylabel('$f_2$')
sv(fig, 'f323.png', 300)

# ---- 32.4-1 four reliability curves ---------------------------------------------
P4 = [('naive Bayes', nb.predict_proba(X_te)[:, 1], 'over-confident'),
      ('AdaBoost', ada.predict_proba(X_te)[:, 1], 'compressed'),
      ('LinearSVC scores, $\\sigma(f)$', sg(svm.decision_function(X_te)), 'under-confident'),
      ('logistic regression', lr.predict_proba(X_te)[:, 1], 'close to the diagonal')]
fig, axs = plt.subplots(2, 2, figsize=(4.9, 4.5), dpi=240, sharex=True, sharey=True)
NUM['four'] = {}
for ax, (nm, p_, verdict) in zip(axs.ravel(), P4):
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.2, ls='--', zorder=1)
    q, t = curve(y_te, p_, 'uniform', 10)
    ax.plot(q, t, '-o', color=BLUE, ms=3.8, lw=1.5, zorder=4)
    ax.fill_between(q, q, t, color=PINK, zorder=0, alpha=0.9)
    hist_ax(ax, p_)
    ax.set_title(nm, fontsize=9, color=NAVY); ax.text(0.04, 0.9, verdict, fontsize=8.6, color=RED, ha='left', va='center', weight='bold')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.2, 1.02); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    NUM['four'][nm] = {'logloss': float(log_loss(y_te, p_)), 'brier': float(brier_score_loss(y_te, p_)), 'pred': q.round(3).tolist(), 'obs': t.round(3).tolist()}
for ax in axs[1]: ax.set_xlabel('predicted probability')
for ax in axs[:, 0]: ax.set_ylabel('observed frequency')
fig.tight_layout(h_pad=0.8, w_pad=0.6); sv(fig, 'f324.png', 470)

# ---- 32.4-2 three calibration maps for naive Bayes probabilities -----------------------
p_ca_nb = nb.predict_proba(X_ca)[:, 1]; p_te_nb = nb.predict_proba(X_te)[:, 1]
grid = np.linspace(1e-4, 1 - 1e-4, 800); maps = {}
for meth in ('sigmoid', 'isotonic', 'temperature'):
    c = CalibratedClassifierCV(FrozenEstimator(nb), method=meth).fit(X_ca, y_ca); cc = c.calibrated_classifiers_[0].calibrators[0]
    if meth == 'temperature': maps[meth] = cc.predict(np.c_[1 - grid, grid])[:, 1]; NUM['maps_beta'] = float(cc.beta_)
    else: maps[meth] = cc.predict(grid)
    if meth == 'sigmoid': NUM['maps_sigmoid'] = [float(cc.a_), float(cc.b_)]
edges = np.quantile(p_te_nb, np.linspace(0, 1, 11)); ib = np.clip(np.searchsorted(edges, p_te_nb, side='right') - 1, 0, 9)
cx = np.array([p_te_nb[ib == k].mean() for k in range(10)]); cy = np.array([y_te[ib == k].mean() for k in range(10)])
fig, ax = plt.subplots(figsize=(4.4, 2.8), dpi=240)
ax.plot(grid, grid, color=GRID, lw=1.5, ls=':', label='no calibration')
ax.plot(grid, maps['sigmoid'], color=GREEN, lw=1.9, label='sigmoid (Platt)')
ax.plot(grid, maps['isotonic'], color=OR, lw=1.7, label='isotonic')
ax.plot(grid, maps['temperature'], color=PUR, lw=1.7, ls='--', label='temperature')
ax.plot(cx, cy, 's', color=GREY, ms=4.5, label='observed (test, deciles)', zorder=6, mfc='white', mew=1.1)
ax.set_xlabel('probability from naive Bayes'); ax.set_ylabel('calibrated probability'); ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.02, 1.02)
ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=8.3, handlelength=2.2, borderaxespad=0.0)
fig.tight_layout(); sv(fig, 'f325.png', 470)

# ---- 32.6-1 flow chart of CalibratedClassifierCV.fit -----------------------------------
fig, ax = new(4.4, 5.2)
ax.set_xlim(0, 11); ax.set_ylim(0, 13.2)
def fb(xc, yc, w, h, t, fc=LB, ec=BLUE, fs=8.3, bold=False):
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc, ec=ec, lw=0.9))
    ax.text(xc, yc, t, ha='center', va='center', fontsize=fs, color=NAVY, linespacing=1.18, weight='bold' if bold else 'normal')
A = lambda p, q: arr(ax, p, q, c=NAVY, lw=0.9, ms=8)
fb(5.5, 12.4, 9.6, 1.2, 'fit(X, y):  estimator, method, cv, ensemble', fc='#E9EDF3', ec=GREY)
fb(5.5, 10.3, 9.6, 1.3, 'ensemble = \'auto\'  becomes  False  for a\nFrozenEstimator and  True  otherwise', fc='#F2E9F7', ec=PUR)
A((5.5, 11.8), (5.5, 10.95))
fb(2.75, 7.6, 4.9, 1.2, 'ensemble = True', fc='#F2E9F7', ec=PUR, bold=True); fb(8.25, 7.6, 4.9, 1.2, 'ensemble = False', fc='#F2E9F7', ec=PUR, bold=True)
A((3.6, 9.65), (2.75, 8.2)); A((7.4, 9.65), (8.25, 8.2))
fb(2.75, 5.5, 4.9, 1.9, 'for each of the cv folds:\nclone the estimator, fit it\non the training part', fc=LB, ec=BLUE)
fb(8.25, 5.5, 4.9, 1.9, 'cross_val_predict:\nout-of-fold scores of\nall n samples', fc=LB, ec=BLUE)
A((2.75, 7.0), (2.75, 6.45)); A((8.25, 7.0), (8.25, 6.45))
fb(2.75, 3.3, 4.9, 1.9, 'score the held-out part,\nfit a calibrator on those\nscores and labels; keep\nthe pair', fc=LB, ec=BLUE)
fb(8.25, 3.3, 4.9, 1.9, 'fit ONE calibrator on the\nout-of-fold scores; refit\nthe estimator on all the\ndata (frozen: no refit)', fc=LB, ec=BLUE)
A((2.75, 4.55), (2.75, 4.25)); A((8.25, 4.55), (8.25, 4.25))
fb(2.75, 1.0, 4.9, 1.4, 'predict_proba:\nmean over the cv pairs', fc='#E3F4EA', ec=GREEN, bold=True)
fb(8.25, 1.0, 4.9, 1.4, 'predict_proba:\ncalibrator(estimator)', fc='#E3F4EA', ec=GREEN, bold=True)
A((2.75, 2.35), (2.75, 1.7)); A((8.25, 2.35), (8.25, 1.7))
sv(fig, 'f326.png', 380)
# ---- 32.9-1 isotonic fits as n grows ---------------------------------------------------
rg = np.random.default_rng(3); truth = lambda x: 2 * sg(1.1 * (x - 5))
fig, axs = plt.subplots(2, 2, figsize=(4.9, 3.7), dpi=240, sharex=True, sharey=True); NUM['gallery'] = {}
xx = np.linspace(0, 10, 300)
for ax, n_ in zip(axs.ravel(), (15, 50, 200, 1000)):
    x_ = np.sort(rg.uniform(0, 10, n_)); y_ = truth(x_) + rg.normal(0, 0.5, n_)
    m_ = IsotonicRegression(out_of_bounds='clip').fit(x_, y_); fv = isotonic_regression(y_); nb_ = len(np.unique(np.round(fv, 12)))
    ax.plot(x_, y_, 'o', color=BLUE, ms=2.6 if n_ > 100 else 3.8, alpha=0.45 if n_ > 100 else 0.9, mec='none', zorder=2)
    ax.plot(xx, truth(xx), color=GREY, lw=1.3, ls='--', zorder=3); ax.plot(xx, m_.predict(xx), color=GREEN, lw=1.9, zorder=4)
    ax.set_title(f'$n$ = {n_}:  {nb_} blocks', fontsize=9, color=NAVY); ax.set_xlim(0, 10); ax.set_ylim(-1.1, 3.2)
    NUM['gallery'][n_] = {'blocks': int(nb_), 'rmse_vs_truth': float(np.sqrt(np.mean((m_.predict(xx) - truth(xx)) ** 2)))}
for ax in axs[1]: ax.set_xlabel('$x$')
for ax in axs[:, 0]: ax.set_ylabel('$y$')
axs[0, 0].text(9.8, -0.6, 'fit', color=GREEN, fontsize=8.5, ha='right', weight='bold'); axs[0, 0].text(9.8, -0.1, 'truth', color=GREY, fontsize=8.5, ha='right')
fig.tight_layout(h_pad=0.8, w_pad=0.6); sv(fig, 'f329a.png', 470)

# ---- 32.9-2 calibration-set size: log loss and Brier of naive Bayes -----------------------
sizes = [50, 100, 200, 500, 1000, 2000, 5000, 10000]; res = {m: {k: [] for k in sizes} for m in ('sigmoid', 'isotonic', 'temperature')}
for n_ in sizes:
    for rep in range(10 if n_ < 10000 else 1):
        idx = np.random.default_rng(100 * rep + n_).choice(len(X_ca), n_, replace=False)
        if len(np.unique(y_ca[idx])) < 2: continue
        for meth in res:
            c = CalibratedClassifierCV(FrozenEstimator(nb), method=meth).fit(X_ca[idx], y_ca[idx]); p_ = c.predict_proba(X_te)[:, 1]
            res[meth][n_].append((log_loss(y_te, p_), brier_score_loss(y_te, p_)))
raw_ll, raw_bs = log_loss(y_te, p_te_nb), brier_score_loss(y_te, p_te_nb)
fig, axs = plt.subplots(1, 2, figsize=(4.9, 2.4), dpi=240); NUM['size'] = {'raw': [raw_ll, raw_bs]}
sty = {'sigmoid': (GREEN, '-', 'o'), 'isotonic': (OR, '-', 's'), 'temperature': (PUR, '--', '^')}
for j, ax in enumerate(axs):
    ax.axhline((raw_ll, raw_bs)[j], color=GREY, lw=1.1, ls=':', zorder=1)
    for meth, (col, ls, mk) in sty.items():
        v = np.array([np.mean([r[j] for r in res[meth][n_]]) for n_ in sizes]); ax.plot(sizes, v, ls=ls, marker=mk, color=col, ms=3.8, lw=1.5, label=meth, zorder=3)
        NUM['size'][meth + ('_ll' if j == 0 else '_bs')] = [round(float(a), 4) for a in v]
    ax.set_xscale('log'); ax.set_xlabel('calibration samples $n$'); ax.grid(color=GRID, lw=0.4)
axs[0].set_yscale('log'); axs[0].set_yticks([0.45, 0.6, 1, 2]); axs[0].set_yticklabels(['0.45', '0.6', '1', '2']); axs[0].set_ylabel('log loss'); axs[1].set_ylabel('Brier score')
axs[0].text(8000, raw_ll * 1.03, 'uncalibrated', color=GREY, fontsize=8.3, va='bottom', ha='right'); axs[1].text(8000, raw_bs - 0.0006, 'uncalibrated', color=GREY, fontsize=8.3, va='top', ha='right'); axs[1].set_ylim(0.1405, 0.1615)
axs[0].legend(loc='upper right', fontsize=8.3, handlelength=2.0, borderaxespad=0.2); axs[0].set_title('log loss (lower is better)', fontsize=9, color=NAVY); axs[1].set_title('Brier score', fontsize=9, color=NAVY)
fig.tight_layout(w_pad=1.0); sv(fig, 'f329b.png', 470)
# ---- 32.11-1 case study: pooled out-of-fold reliability --------------------------------------
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
Xb, yb = load_breast_cancer(return_X_y=True); skf = StratifiedKFold(5, shuffle=True, random_state=0)
def oof(est): return cross_val_predict(est, Xb, yb, cv=skf, method='predict_proba')[:, 1]
cv_ = lambda e, m: CalibratedClassifierCV(e, method=m, cv=5, ensemble=False)
PC = {'NB raw': oof(GaussianNB()), 'NB sigmoid': oof(cv_(GaussianNB(), 'sigmoid')), 'NB temperature': oof(cv_(GaussianNB(), 'temperature')),
      'Ada raw': oof(AdaBoostClassifier(n_estimators=200, random_state=0)), 'Ada sigmoid': oof(cv_(AdaBoostClassifier(n_estimators=200, random_state=0), 'sigmoid'))}
NUM['case_oof'] = {k: {'logloss': float(log_loss(yb, v)), 'brier': float(brier_score_loss(yb, v))} for k, v in PC.items()}
fig, axs = plt.subplots(1, 2, figsize=(4.9, 2.45), dpi=240, sharey=True)
for ax, ttl, items in ((axs[0], 'naive Bayes', (('NB raw', BLUE, 'o', '-', 'raw'), ('NB sigmoid', GREEN, 's', '-', 'sigmoid'), ('NB temperature', PUR, '^', '--', 'temperature'))),
                       (axs[1], 'AdaBoost, 200 stumps', (('Ada raw', BLUE, 'o', '-', 'raw'), ('Ada sigmoid', GREEN, 's', '-', 'sigmoid')))):
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.2, ls='--', zorder=1)
    for k, col, mk, ls, lab in items:
        q, t = curve(yb, PC[k], 'quantile', 6); ax.plot(q, t, ls=ls, marker=mk, color=col, ms=4, lw=1.4, label=lab, zorder=4)
    ax.set_title(ttl, fontsize=9, color=NAVY); ax.set_xlabel('predicted probability'); ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.legend(loc='lower right', fontsize=8.3, handlelength=2.0, borderaxespad=0.2)
axs[0].set_ylabel('observed frequency (benign)')
fig.tight_layout(w_pad=1.0); sv(fig, 'f3211.png', 470)
json.dump({'NUM': NUM, 'CHK': CHK}, open('verify/fig_numbers.json', 'w'), indent=1)
json.dump({'NUM': NUM, 'CHK': CHK}, open('verify/fig_numbers.json', 'w'), indent=1)
