# All figures of Chapter 29. Run: python3 fig/figs.py [names]  (PNGs into ch29/, numbers into verify/fig_numbers.json)
import sys, os, json, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../lib'); sys.path.insert(0, HERE + '/../code'); os.chdir(HERE + '/..')
os.environ['OMP_NUM_THREADS'] = '1'
warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer, make_classification, make_moons
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from condorcet import majority
plt.rcParams.update({'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'axes.labelsize': 9, 'legend.fontsize': 8.5})
NUM = {}; CHK = {}
ONLY = set(sys.argv[1:])
def want(k): return (not ONLY) or (k in ONLY)
def sv(fig, name, printed_px):
    fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig, name); from PIL import Image
    with Image.open(name) as im: w = im.size[0] / 240
    CHK[name] = {'fig_in': round(w, 2), 'printed_in': round(printed_px / 96, 2), 'min_font_design': round(fs, 1), 'min_font_printed': round(fs * printed_px / 96 / w, 1)}
def grid(ax): ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
LO = '#FDEBD0'

# ---------------------------------------------------------------- f290  opener: voting and stacking
if want('f290'):
    fig, ax = plt.subplots(figsize=(4.8, 4.1)); ax.set_xlim(0, 10); ax.set_ylim(0, 9.2); ax.axis('off')
    xs = [0.1, 2.3, 4.5]; pv = [0.95, 0.30, 0.40]
    ax.text(0.0, 9.15, '1  Voting: every member answers, the answers are counted or averaged', fontsize=9, color=NAVY, weight='bold', va='top')
    for x0, nm, p in zip(xs, 'ABC', pv):
        ax.add_patch(FancyBboxPatch((x0, 6.15), 1.95, 2.35, boxstyle='round,pad=0.02,rounding_size=0.12', fc=LB, ec=BLUE, lw=0.9))
        ax.text(x0 + 0.97, 8.35, 'doctor ' + nm, ha='center', va='top', fontsize=8.8, color=NAVY)
        ax.add_patch(Rectangle((x0 + 0.3, 6.4), 0.55, 1.5, fc='white', ec=GREY, lw=0.6))
        ax.add_patch(Rectangle((x0 + 0.3, 6.4), 0.55, 1.5 * p, fc=BLUE if p > 0.5 else OR, ec='none'))
        ax.plot([x0 + 0.22, x0 + 0.93], [6.4 + 0.75] * 2, color=NAVY, lw=0.7, ls=':')
        ax.text(x0 + 1.02, 6.4 + 1.5 * p, f'{p:.2f}', fontsize=8.8, color=NAVY, va='center')
    ax.text(7.05, 7.35, 'P(disease)\nfor one patient,\nbars from 0 to 1,\ndotted line at \u00bd', fontsize=8.5, color=GREY, va='center')
    for x0 in xs: arr(ax, (x0 + 0.97, 6.05), (x0 + 0.97, 5.35), c=GREY, lw=0.8, ms=6)
    ax.add_patch(FancyBboxPatch((0.1, 4.1), 7.35, 1.2, boxstyle='round,pad=0.02,rounding_size=0.12', fc='#F6F7F9', ec=GREY, lw=0.9))
    ax.plot([3.775, 3.775], [4.2, 5.2], color=GREY, lw=0.7)
    ax.text(1.95, 4.7, 'hard vote: 1 of 3 above \u00bd\n$\\to$ class 0', ha='center', va='center', fontsize=8.5, color=NAVY)
    ax.text(5.6, 4.7, 'soft vote: average 0.55\n$\\to$ class 1', ha='center', va='center', fontsize=8.5, color=OR, weight='bold')
    ax.plot([0, 10], [3.55, 3.55], color=GRID, lw=0.8)
    ax.text(0.0, 3.35, '2  Stacking: a final model learns how far to trust each member', fontsize=9, color=NAVY, weight='bold', va='top')
    for x0, k in zip(xs, '123'):
        box(ax, x0, 1.95, 1.95, 0.8, f'member {k}', fc=LB, ec=BLUE, fs=8.8)
        ax.add_patch(Rectangle((x0 + 0.35, 0.35), 1.25, 1.3, fc='white', ec=GREEN, lw=0.9))
        for yy in (0.78, 1.21): ax.plot([x0 + 0.35, x0 + 1.6], [yy, yy], color=GREEN, lw=0.5)
        ax.text(x0 + 0.97, 1.43, f'$z_{k}$', ha='center', va='center', fontsize=9, color=GREEN)
        arr(ax, (x0 + 0.97, 1.9), (x0 + 0.97, 1.7), c=GREEN, lw=0.9, ms=5)
    ax.text(0.0, 0.0, 'columns of out-of-fold answers, one row per training case', fontsize=8.3, color=GREEN, va='center')
    arr(ax, (6.65, 1.0), (7.15, 1.0), c=GREEN, lw=1.0, ms=7)
    box(ax, 7.2, 0.35, 2.75, 1.3, 'final model\nlearns $w_1, w_2, w_3$', fc='#DFF1E6', ec=GREEN, fs=8.5)
    ax.set_ylim(-0.5, 9.2)
    fig.tight_layout(); sv(fig, 'f290.png', 450)

# ---------------------------------------------------------------- f292  Condorcet: independent and correlated voters
if want('f292'):
    ms = np.arange(1, 52, 2)
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.5), sharey=True)
    ax = axs[0]
    for p, col, ls in ((0.7, GREEN, '-'), (0.6, BLUE, '-'), (0.55, PUR, '-'), (0.45, RED, '--')):
        ax.plot(ms, [majority(m, p) for m in ms], color=col, ls=ls, lw=1.5)
        yv = majority(51, p); ax.text(52.5, yv + (0.0 if p != 0.7 else 0.0), f'{p}', fontsize=8.5, color=col, va='center')
    ax.axhline(0.5, color=GREY, lw=0.7, ls=':'); ax.set_xlim(0, 58); ax.set_ylim(0.1, 1.04)
    ax.set_xlabel('number of voters $M$'); ax.set_ylabel('majority is right'); ax.set_title('independent voters, accuracy $p$', fontsize=8.8, color=NAVY); grid(ax)
    ax = axs[1]
    for rho, col, ls in ((0.0, GREEN, '-'), (0.1, BLUE, '-'), (0.3, PUR, '-'), (0.6, RED, '-')):
        ax.plot(ms, [majority(m, 0.7, rho) for m in ms], color=col, ls=ls, lw=1.5)
        ax.text(52.5, majority(51, 0.7, rho), f'{rho:g}', fontsize=8.5, color=col, va='center')
    ax.axhline(0.7, color=GREY, lw=0.7, ls=':'); ax.set_xlim(0, 58)
    ax.set_xlabel('number of voters $M$'); ax.set_title('$p=0.7$, correlation $\\rho$', fontsize=8.8, color=NAVY); grid(ax)
    fig.tight_layout(w_pad=0.8); sv(fig, 'f292.png', 450)
    NUM['f292'] = {'p0.7_M51': float(majority(51, 0.7)), 'rho0.3_M51': float(majority(51, 0.7, 0.3))}

# ---------------------------------------------------------------- f293  decision regions of members and committees on noisy moons
def moons_models():
    X, y = make_moons(1200, noise=0.35, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=200, random_state=0, stratify=y)
    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    est = [("line", LogisticRegression()), ("near", KNeighborsClassifier(15)), ("tree", tree)]
    ms = {'logistic regression': est[0][1], '15-nearest neighbours': est[1][1], 'tree, depth 3': est[2][1],
          'hard vote': VotingClassifier(est), 'soft vote': VotingClassifier(est, voting='soft'), 'stacking': StackingClassifier(est)}
    for m in ms.values(): m.fit(Xtr, ytr)
    return Xtr, ytr, Xte, yte, ms
if want('f293'):
    Xtr, ytr, Xte, yte, ms = moons_models()
    gx, gy = np.meshgrid(np.linspace(-2, 3, 220), np.linspace(-1.6, 2.2, 190)); G = np.c_[gx.ravel(), gy.ravel()]
    fig, axs = plt.subplots(2, 3, figsize=(4.8, 3.5), sharex=True, sharey=True); res = {}
    cm = ListedColormap([LB, LO])
    for ax, (nm, m) in zip(axs.ravel(), ms.items()):
        z = m.predict(G).reshape(gx.shape); ax.contourf(gx, gy, z, levels=[-0.5, 0.5, 1.5], cmap=cm, zorder=0)
        ax.contour(gx, gy, z, levels=[0.5], colors=[GREEN], linewidths=1.3, zorder=1)
        for cls, col, mk in ((0, BLUE, 'o'), (1, OR, 's')): ax.scatter(*Xtr[ytr == cls].T, s=5, color=col, marker=mk, lw=0, zorder=3)
        acc = m.score(Xte, yte); res[nm] = round(float(acc), 4)
        ax.set_title(f'{nm}\ntest {acc:.3f}', fontsize=8.5, color=NAVY); ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ax.spines.values(): sp_.set_visible(True); sp_.set_color(GRID)
    fig.tight_layout(h_pad=0.5, w_pad=0.3); sv(fig, 'f293.png', 450); NUM['f293'] = res

# ---------------------------------------------------------------- f296  flow chart of StackingClassifier.fit
if want('f296'):
    fig, ax = plt.subplots(figsize=(5.0, 2.7)); ax.set_xlim(0, 20); ax.set_ylim(0, 10.8); ax.axis('off')
    def bx(x, y, w, h, t, fc, ec, fs=8.6): box(ax, x, y, w, h, t, fc=fc, ec=ec, fs=fs)
    bx(0.1, 6.0, 4.9, 4.4, '1  validate X, y;\nencode the\nlabels', LB, BLUE)
    bx(6.1, 6.0, 7.5, 4.4, '2  for each member $m$:\ncross_val_predict with its\nstack_method over $K$ folds\n$\\to$ column $z_m$ (Eq. 29.4-1)', '#DFF1E6', GREEN)
    bx(14.7, 6.0, 5.2, 4.4, '3  refit every member\non all the rows\n$\\to$ estimators_', '#F1EAF7', PUR)
    bx(14.7, 0.4, 5.2, 4.4, '4  meta matrix\n$Z=[z_1\\ldots z_M]$\n(passthrough: [Z  X])', LB, BLUE)
    bx(6.1, 0.4, 7.5, 4.4, '5  fit final_estimator_\non (Z, y)  (Eq. 29.4-2)', '#DFF1E6', GREEN)
    bx(0.1, 0.4, 4.9, 4.4, 'predict:\nmembers $\\to$\ntransform $\\to$\nfinal_estimator_', '#FFFFFF', GREY)
    arr(ax, (5.0, 8.2), (6.1, 8.2), c=NAVY, lw=1.0); arr(ax, (13.6, 8.2), (14.7, 8.2), c=NAVY, lw=1.0)
    arr(ax, (17.3, 6.0), (17.3, 4.8), c=NAVY, lw=1.0)
    arr(ax, (14.7, 2.6), (13.6, 2.6), c=NAVY, lw=1.0); arr(ax, (6.1, 2.6), (5.0, 2.6), c=NAVY, lw=1.0, ls=':')
    fig.tight_layout(); sv(fig, 'f296.png', 450)

# ---------------------------------------------------------------- f297  leakage: in-sample level-1 matrix
if want('f297'):
    X, y = make_classification(600, 10, n_informative=3, n_redundant=0, flip_y=0.15, random_state=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
    def members(): return [("lr", make_pipeline(StandardScaler(), LogisticRegression())), ("rf", RandomForestClassifier(100, random_state=0)), ("nn1", KNeighborsClassifier(1))]
    fitted = [(n, m.fit(Xtr, ytr)) for n, m in members()]
    honest = StackingClassifier(members()).fit(Xtr, ytr); leaky = StackingClassifier(fitted, cv='prefit').fit(Xtr, ytr)
    tr = [m.score(Xtr, ytr) for _, m in fitted]; te = [m.score(Xte, yte) for _, m in fitted]
    wh, wl = honest.final_estimator_.coef_.ravel(), leaky.final_estimator_.coef_.ravel()
    fig, axs = plt.subplots(1, 3, figsize=(4.8, 2.3), gridspec_kw={'width_ratios': [1, 1, 1.05]})
    labs = ['LR', 'RF', '1-NN']; xx = np.arange(3)
    ax = axs[0]; ax.bar(xx - 0.2, tr, 0.38, color=BLUE, label='train', zorder=3); ax.bar(xx + 0.2, te, 0.38, color=GREY, label='test', zorder=3)
    ax.set_xticks(xx); ax.set_xticklabels(labs, fontsize=8.5); ax.set_ylim(0.5, 1.12); ax.set_ylabel('accuracy'); ax.set_title('the members', fontsize=8.8, color=NAVY); ax.legend(loc='upper left', fontsize=8, ncol=2, columnspacing=0.8, handlelength=1.0, borderaxespad=0.1); grid(ax)
    ax = axs[1]; ax.bar(xx - 0.2, wh, 0.38, color=GREEN, label='out-of-fold', zorder=3); ax.bar(xx + 0.2, wl, 0.38, color=RED, label='in-sample', zorder=3, hatch='///', ec='white')
    ax.set_xticks(xx); ax.set_xticklabels(labs, fontsize=8.5); ax.set_ylim(0, 6.6); ax.set_ylabel('weight $w_m$'); ax.set_title('level-1 weights', fontsize=8.8, color=NAVY); ax.legend(loc='upper left', fontsize=8, handlelength=1.0, borderaxespad=0.1); grid(ax)
    ax = axs[2]; sc_ = [honest.score(Xte, yte), leaky.score(Xte, yte)]
    ax.bar([0, 1], sc_, 0.55, color=[GREEN, RED], zorder=3); ax.axhline(te[0], color=BLUE, lw=1.0, ls='--'); ax.text(1.45, te[0] - 0.025, 'LR', fontsize=8.5, color=BLUE, ha='right', va='top')
    for i, v in enumerate(sc_): ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=8.5, color=NAVY)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['out-of-\nfold', 'in-\nsample'], fontsize=8.3); ax.set_ylim(0.5, 0.9); ax.set_title('stacked, test', fontsize=8.8, color=NAVY); grid(ax)
    fig.tight_layout(w_pad=0.8); sv(fig, 'f297.png', 450)
    NUM['f297'] = {'train': [round(v, 4) for v in tr], 'test': [round(v, 4) for v in te], 'w_oof': wh.round(2).tolist(), 'w_in': wl.round(2).tolist(), 'stack_test': [round(float(v), 4) for v in sc_]}

# ---------------------------------------------------------------- f2911  case study: accuracies and the errors
if want('f2911'):
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    sc = StandardScaler
    est = [("lr", make_pipeline(sc(), LogisticRegression())),
           ("svc", make_pipeline(sc(), CalibratedClassifierCV(SVC(), method="sigmoid", cv=5, ensemble=False))),
           ("rf", RandomForestClassifier(100, random_state=0)), ("knn", make_pipeline(sc(), KNeighborsClassifier()))]
    committees = {'hard vote': VotingClassifier(est), 'soft vote': VotingClassifier(est, voting='soft'), 'stacking': StackingClassifier(est)}
    allm = [('LR', est[0][1]), ('SVC', est[1][1]), ('RF', est[2][1]), ('kNN', est[3][1])] + list(committees.items())
    cv = {}; pred = {}
    for nm, m in allm:
        cv[nm] = float(cross_val_score(m, Xtr, ytr, cv=5).mean()); m.fit(Xtr, ytr); pred[nm] = m.predict(Xte)
    names = [n for n, _ in allm]; wrong = np.array([pred[n] != yte for n in names])
    te = [1 - w.mean() for w in wrong]
    fig, axs = plt.subplots(1, 2, figsize=(5.0, 3.1), gridspec_kw={'width_ratios': [1.0, 1.0]})
    ax = axs[0]; yy = np.arange(len(names))[::-1]
    for k, (n, yv) in enumerate(zip(names, yy)):
        col = BLUE if k < 4 else GREEN
        ax.plot([cv[n], te[k]], [yv, yv], color=GRID, lw=1.0, zorder=1)
        ax.scatter(cv[n], yv, s=34, facecolor='white', edgecolor=col, lw=1.3, zorder=3, marker='o')
        ax.scatter(te[k], yv, s=34, color=col, zorder=4, marker='s')
    ax.axhline(3.5 if False else 2.5, color=GRID, lw=0.8)
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=8.8); ax.set_xlim(0.935, 0.992); ax.set_xlabel('accuracy'); ax.set_xticks([0.94, 0.96, 0.98]); grid(ax)
    ax.scatter([], [], s=34, facecolor='white', edgecolor=GREY, lw=1.3, label='5-fold, training part'); ax.scatter([], [], s=34, color=GREY, marker='s', label='test, 143 patients')
    ax.legend(loc='lower left', fontsize=8, handletextpad=0.2, borderaxespad=0.1, bbox_to_anchor=(0.0, 0.0)); ax.set_ylim(-2.2, len(names) - 0.4)
    ax = axs[1]
    nwr = wrong[:4].sum(0); idx = np.where(wrong.any(0))[0]; idx = idx[np.lexsort((np.arange(len(idx)), -nwr[idx]))]
    for r, j in enumerate(idx):
        for c_, n in enumerate(names):
            if wrong[c_, j]: ax.add_patch(Rectangle((c_ - 0.36, -r - 0.36), 0.72, 0.72, fc=RED, ec='none', zorder=3))
            else: ax.scatter(c_, -r, s=4, color=GRID, zorder=2)
    ax.axvline(3.5, color=GREY, lw=0.8)
    ax.set_xlim(-0.6, 6.6); ax.set_ylim(-len(idx) + 0.4, 0.8); ax.set_xticks(range(7)); ax.set_xticklabels(names, fontsize=8.0, rotation=90)
    ax.set_yticks(range(0, -len(idx), -1)); ax.set_yticklabels([str(int(yte[j]) and 'B' or 'M') for j in idx], fontsize=8)
    ax.set_ylabel('patient (M malignant, B benign)', fontsize=8.5); ax.xaxis.tick_top(); ax.spines['top'].set_visible(False); ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False); ax.tick_params(length=0)
    fig.tight_layout(w_pad=1.2); sv(fig, 'f2911.png', 470)
    NUM['f2911'] = {'cv': {k: round(v, 4) for k, v in cv.items()}, 'test': {n: round(float(t), 4) for n, t in zip(names, te)}, 'errors': {n: int(w.sum()) for n, w in zip(names, wrong)}, 'patients_with_an_error': int(len(idx))}
# ---------------------------------------------------------------- f298  regression committees: R2 and the ambiguity decomposition
if want('f298'):
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor, VotingRegressor
    from sklearn.linear_model import LinearRegression
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    est = [("ols", LinearRegression()), ("forest", RandomForestRegressor(n_estimators=300, max_features=0.33, min_samples_leaf=5, random_state=0)),
           ("boost", GradientBoostingRegressor(random_state=0))]
    vote = VotingRegressor(est).fit(Xtr, ytr); stack = StackingRegressor(est).fit(Xtr, ytr)
    F = vote.transform(Xte); fbar = F.mean(1)
    mses = ((F - yte[:, None]) ** 2).mean(0); lhs = float(np.mean((fbar - yte) ** 2)); spread = float(np.mean((F - fbar[:, None]) ** 2))
    r2 = [m.score(Xte, yte) for m in vote.estimators_] + [vote.score(Xte, yte), stack.score(Xte, yte)]
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.4), gridspec_kw={'width_ratios': [1.15, 1]})
    ax = axs[0]; labs = ['OLS', 'forest', 'boost', 'average', 'stack']; xx = np.arange(5)
    ax.bar(xx, r2, 0.62, color=[BLUE] * 3 + [GREEN] * 2, zorder=3)
    for i, v in enumerate(r2): ax.text(i, v + 0.008, f'{v:.3f}', ha='center', fontsize=8.0, color=NAVY)
    ax.set_xticks(xx); ax.set_xticklabels(labs, fontsize=8.0, rotation=35, ha='right'); ax.set_ylim(0, 0.44); ax.set_ylabel('test $R^2$'); ax.set_title('members and committees', fontsize=8.8, color=NAVY); grid(ax)
    ax = axs[1]
    ax.bar([0], [mses.mean()], 0.55, color=BLUE, zorder=3); ax.bar([1], [lhs], 0.55, color=GREEN, zorder=3); ax.bar([1], [spread], 0.55, bottom=[lhs], color=PUR, zorder=3, hatch='///', ec='white')
    ax.axhline(mses[0], color=GREY, lw=1.0, ls='--'); ax.text(0.5, mses[0] + 60, 'OLS', fontsize=8.0, color=GREY, ha='center', va='bottom')
    ax.text(1, lhs / 2, f'{lhs:.0f}', ha='center', va='center', fontsize=8.0, color='white'); ax.text(1.0, lhs + spread + 50, f'+{spread:.0f}', ha='center', fontsize=8.0, color=PUR)
    ax.text(0, mses.mean() / 2, f'{mses.mean():.0f}', ha='center', va='center', fontsize=8.0, color='white')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['mean of the\nmembers', 'average\n+ spread'], fontsize=8.0); ax.set_ylim(0, 4200); ax.set_ylabel('squared error'); ax.set_title('Eq. 29.3-6', fontsize=8.8, color=NAVY); grid(ax)
    fig.tight_layout(w_pad=0.8); sv(fig, 'f298.png', 450)
    NUM['f298'] = {'test_r2': dict(zip(labs, [round(float(v), 4) for v in r2])), 'member_mse': [round(float(v), 1) for v in mses], 'mean_member_mse': round(float(mses.mean()), 1), 'mse_of_average': round(lhs, 1), 'spread': round(spread, 1)}

if not ONLY:
    with open('verify/fig_numbers.json', 'w') as fh: json.dump(NUM, fh, indent=1)
    with open('verify/fig_textsize.json', 'w') as fh: json.dump(CHK, fh, indent=1)
