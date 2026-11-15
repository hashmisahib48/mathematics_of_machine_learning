# Problem figures of Chapter 29. Run: python3 fig/probs.py  (answers -> verify/problem_answers.json)
import sys, os, json, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../lib'); sys.path.insert(0, HERE + '/../code'); os.chdir(HERE + '/..')
os.environ['OMP_NUM_THREADS'] = '1'
warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from scipy.stats import binom
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer, load_wine, make_classification
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_predict, cross_val_score, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from condorcet import majority
plt.rcParams.update({'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.labelsize': 9.5, 'legend.fontsize': 9})
A = {}; CHK = {}
def small(w=2.4, h=1.05): return plt.subplots(figsize=(w, h), dpi=240)
def sv(fig, name):
    fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()); save(fig, name)
    from PIL import Image
    with Image.open(name) as im: w, h = im.size
    pw = min(1.55 * 135, 235 * w / h); CHK[name] = round(fs * pw / 96 / (w / 240), 1)
def grid(ax): ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
r4 = lambda v: [round(float(u), 4) for u in v]

# 29A.1 five independent voters
p = 0.8; pm = binom.pmf(range(6), 5, p)
A['29A.1'] = {'M5': round(float(majority(5, p)), 5), 'M3': round(float(majority(3, p)), 5), 'M1': p, 'pmf': r4(pm)}
fig, ax = small(); ax.bar(range(6), pm, color=[GREY if k < 3 else GREEN for k in range(6)], width=0.65, zorder=3)
ax.set_xlabel('voters that are right'); ax.set_ylabel('probability'); ax.set_xticks(range(6)); grid(ax); sv(fig, 'p_29A1.png')

# 29A.2 hard against soft
P = np.array([0.60, 0.55, 0.05]); w = np.array([1, 1, 3.])
ll = -np.log(P); soft = P.mean(); softw = P @ w / w.sum()
A['29A.2'] = {'hard_votes_for_1': int((P > 0.5).sum()), 'soft': round(float(soft), 4), 'soft_w': round(float(softw), 4), 'member_log_loss': r4(ll), 'mean_member_log_loss': round(float(ll.mean()), 4), 'soft_log_loss': round(float(-np.log(soft)), 4)}
fig, ax = small(); ax.bar(range(3), P, color=BLUE, width=0.6, zorder=3); ax.axhline(0.5, color=GREY, lw=1.0, ls=':'); ax.axhline(soft, color=GREEN, lw=1.4, ls='--')
ax.set_xticks(range(3)); ax.set_xticklabels(['1', '2', '3']); ax.set_xlabel('member'); ax.set_ylabel('$P(y=1)$'); ax.set_ylim(0, 0.8); grid(ax); sv(fig, 'p_29A2.png')

# 29A.3 the ambiguity identity
f = np.array([4., 7., 10.]); yv = 6.0; fb = f.mean()
A['29A.3'] = {'mean': float(fb), 'mse_of_average': float((fb - yv) ** 2), 'mean_member_mse': float(np.mean((f - yv) ** 2)), 'spread': float(np.mean((f - fb) ** 2))}
fig, ax = small(2.4, 0.6); ax.scatter(f, np.zeros(3), s=40, color=BLUE, zorder=4, edgecolor='white'); ax.scatter([fb], [0], s=60, marker='D', color=GREEN, zorder=5, edgecolor='white'); ax.scatter([yv], [0], s=60, marker='*', color=RED, zorder=5)
ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_ylim(-0.5, 0.5); ax.set_xlim(2.5, 11.5); ax.set_xticks([4, 6, 8, 10]); ax.set_xlabel('value'); sv(fig, 'p_29A3.png')

# 29A.4 width of the level-1 matrix and number of fits
class Count(LogisticRegression):
    n_fits = 0
    def fit(self, X, y, **kw):
        Count.n_fits += 1
        return super().fit(X, y, **kw)
CNT = lambda: Count(max_iter=5000)
Xw, yw = load_wine(return_X_y=True)
est = [('lr', make_pipeline(StandardScaler(), LogisticRegression())), ('svc', make_pipeline(StandardScaler(), SVC())), ('tree', DecisionTreeClassifier(max_depth=3, random_state=0))]
st = StackingClassifier(est, cv=7).fit(Xw, yw); width = st.transform(Xw).shape[1]
stp = StackingClassifier(est, cv=7, passthrough=True, final_estimator=make_pipeline(StandardScaler(), LogisticRegression())).fit(Xw, yw)
Count.n_fits = 0
StackingClassifier([('a', CNT()), ('b', CNT()), ('c', CNT())], cv=7).fit(StandardScaler().fit_transform(Xw), yw)
A['29A.4'] = {'stack_method_': st.stack_method_, 'width': int(width), 'width_passthrough': int(stp.transform(Xw).shape[1]), 'fits_of_the_three_members': Count.n_fits, 'fits_with_final': Count.n_fits + 1, 'n_features': int(Xw.shape[1]), 'classes': 3}
fig, ax = small(2.4, 0.85); ax.bar([0, 1], [width, 13], width=0.55, color=[GREEN, GREY], zorder=3); ax.set_xticks([0, 1]); ax.set_xticklabels(['$\\mathbf{Z}$', '$\\mathbf{X}$'], fontsize=10); ax.set_ylabel('columns'); ax.set_ylim(0, 16); grid(ax); sv(fig, 'p_29A4.png')

# data of the cancer problems
Xc, yc = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.25, random_state=0, stratify=yc)
sc = StandardScaler
def four():
    return [('lr', make_pipeline(sc(), LogisticRegression())),
            ('svc', make_pipeline(sc(), CalibratedClassifierCV(SVC(), method='sigmoid', cv=5, ensemble=False))),
            ('rf', RandomForestClassifier(100, random_state=0)), ('knn', make_pipeline(sc(), KNeighborsClassifier()))]

# 29B.2 a fifth voter
from sklearn.ensemble import GradientBoostingClassifier
h4 = VotingClassifier(four()).fit(Xtr, ytr); h5 = VotingClassifier(four() + [('gb', GradientBoostingClassifier(random_state=0))]).fit(Xtr, ytr)
p4, p5 = h4.predict(Xte), h5.predict(Xte); gb = h5.estimators_[4].predict(Xte)
A['29B.2'] = {'errors_4': int((p4 != yte).sum()), 'errors_5': int((p5 != yte).sum()), 'differ': int((p4 != p5).sum()), 'gb_errors': int((gb != yte).sum())}
fig, ax = small(2.4, 0.85); ax.bar([0, 1], [A['29B.2']['errors_4'], A['29B.2']['errors_5']], width=0.55, color=[GREY, GREEN], zorder=3); ax.set_xticks([0, 1]); ax.set_xticklabels(['4 voters', '5 voters'], fontsize=8.8); ax.set_ylabel('test errors'); ax.set_ylim(0, 9); grid(ax); sv(fig, 'p_29B2.png')

# 29B.1 correlated voters by simulation
rng = np.random.default_rng(0); n = 400000; pp = 0.7; M = 5; sim = {}; form = {}
for rho in (0.1, 0.3):
    s_ = (1 - rho) / rho; q = rng.beta(pp * s_, (1 - pp) * s_, n); right = rng.random((n, M)) < q[:, None]
    sim[str(rho)] = round(float((right.sum(1) > M / 2).mean()), 4); form[str(rho)] = round(float(majority(M, pp, rho)), 4)
A['29B.1'] = {'simulation': sim, 'formula': form, 'independent': round(float(majority(M, pp)), 4)}
fig, ax = small(2.4, 0.85); xs_ = np.arange(3); vals = [majority(M, pp), form['0.1'], form['0.3']]
ax.bar(xs_, vals, width=0.55, color=[GREEN, BLUE, PUR], zorder=3); ax.set_xticks(xs_); ax.set_xticklabels(['0', '0.1', '0.3']); ax.set_xlabel('correlation $\\rho$'); ax.set_ylabel('majority right'); ax.set_ylim(0.6, 0.9); grid(ax); sv(fig, 'p_29B1.png')

# 29B.3 the final estimator
def three(): return [e for e in four() if e[0] != 'svc']
finals = {'LR': LogisticRegression(), 'LR, C=0.01': LogisticRegression(C=0.01), 'tree': DecisionTreeClassifier(random_state=0), 'forest': RandomForestClassifier(100, random_state=0)}
res = {}
for nm, fe in finals.items():
    m = StackingClassifier(three(), fe)
    cv = cross_val_score(m, Xtr, ytr, cv=5).mean(); m.fit(Xtr, ytr); res[nm] = {'cv': round(float(cv), 4), 'test': round(float(m.score(Xte, yte)), 4)}
A['29B.3'] = res
fig, ax = small(2.4, 0.85); xs_ = np.arange(4); ax.bar(xs_ - 0.2, [res[k]['cv'] for k in finals], 0.38, color=BLUE, zorder=3, label='5-fold'); ax.bar(xs_ + 0.2, [res[k]['test'] for k in finals], 0.38, color=GREY, zorder=3, label='test')
ax.set_xticks(xs_); ax.set_xticklabels(['LR', 'LR\n$C$=.01', 'tree', 'forest'], fontsize=8.5); ax.set_ylim(0.9, 1.035); ax.set_yticks([0.9, 0.95, 1.0]); ax.set_ylabel('accuracy'); ax.legend(loc='upper center', fontsize=8.5, ncol=2, borderaxespad=0.1, columnspacing=0.8, handlelength=1.0); grid(ax); sv(fig, 'p_29B3.png')

# 29C.1 an over-confident member
knn = four()[3][1].fit(Xtr, ytr); lr = four()[0][1].fit(Xtr, ytr)
pk = knn.predict_proba(Xte)[np.arange(len(yte)), yte]; pl = lr.predict_proba(Xte)[np.arange(len(yte)), yte]
A['29C.1'] = {'knn_values_of_P_true': sorted({round(float(v), 2) for v in pk}), 'knn_patients_with_P_true_0': int((pk == 0).sum()), 'lr_patients_P_true_below_0.1': int((pl < 0.1).sum()), 'knn_logloss': round(float(log_loss(yte, knn.predict_proba(Xte))), 4), 'lr_logloss': round(float(log_loss(yte, lr.predict_proba(Xte))), 4), 'clip_cost_each': round(float(-np.log(1e-15) / len(yte)), 4)}
fig, ax = small(); bins = np.linspace(0, 1, 6); kc = np.histogram(pk, bins=[-0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01])[0]; lc = np.histogram(pl, bins=[-0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01])[0]
xs_ = np.arange(6); ax.bar(xs_ - 0.2, kc, 0.38, color=OR, label='kNN', zorder=3); ax.bar(xs_ + 0.2, lc, 0.38, color=BLUE, label='LR', zorder=3)
ax.set_xticks(xs_); ax.set_xticklabels(['0', '.2', '.4', '.6', '.8', '1'], fontsize=8.8); ax.set_xlabel('probability of the true class'); ax.set_ylabel('patients'); ax.set_yscale('log'); ax.set_ylim(0.8, 400); ax.legend(loc='upper center', fontsize=8.8, ncol=2, borderaxespad=0.1, columnspacing=0.8, handlelength=1.0); grid(ax); sv(fig, 'p_29C1.png')

# 29C.2 why out-of-fold
Xs, ys = make_classification(600, 10, n_informative=3, n_redundant=0, flip_y=0.15, random_state=1)
Xa, Xb, ya, yb = train_test_split(Xs, ys, test_size=0.5, random_state=0, stratify=ys)
mem = [('LR', make_pipeline(sc(), LogisticRegression())), ('RF', RandomForestClassifier(100, random_state=0)), ('1-NN', KNeighborsClassifier(1))]
ins, oof = [], []
for nm, m in mem:
    oof.append(float((cross_val_predict(m, Xa, ya, cv=StratifiedKFold(5)) == ya).mean())); ins.append(float(m.fit(Xa, ya).score(Xa, ya)))
A['29C.2'] = {'in_sample': r4(ins), 'out_of_fold': r4(oof)}
fig, ax = small(); xs_ = np.arange(3); ax.bar(xs_ - 0.2, ins, 0.38, color=RED, hatch='///', ec='white', zorder=3, label='in-sample'); ax.bar(xs_ + 0.2, oof, 0.38, color=GREEN, zorder=3, label='out-of-fold')
ax.set_xticks(xs_); ax.set_xticklabels([n for n, _ in mem], fontsize=8.8); ax.set_ylim(0.5, 1.2); ax.set_yticks([0.6, 0.8, 1.0]); ax.set_ylabel('accuracy'); ax.legend(loc='upper left', fontsize=8.3, ncol=2, borderaxespad=0.1, columnspacing=0.8, handlelength=1.0); grid(ax); sv(fig, 'p_29C2.png')

with open('verify/problem_answers.json', 'w') as fh: json.dump(A, fh, indent=1)
with open('verify/probfig_textsize.json', 'w') as fh: json.dump(CHK, fh, indent=1)
print({k: v for k, v in CHK.items() if v < 7.5}); print(json.dumps(A))
