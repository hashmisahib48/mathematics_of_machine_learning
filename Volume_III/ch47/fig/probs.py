# Problem figures and answers of Chapter 47. Run: python3 fig/probs.py
from common import *
from scipy.special import gammaln
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
A = {}
plt.rcParams.update({'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.labelsize': 9, 'legend.fontsize': 9})
def small(w=2.4, h=0.85): return plt.subplots(figsize=(w, h), dpi=240)

# ---- 47A.1  tophat and Epanechnikov at x = 3
d = np.array([1., 2., 4., 9.]); h = 2.5; x0 = 3.0; u = np.abs(x0 - d) / h
top = np.where(u < 1, 0.5, 0.0).sum() / (4 * h); epa = np.where(u < 1, 0.75 * (1 - u ** 2), 0.0).sum() / (4 * h)
sk = {k: float(np.exp(KernelDensity(bandwidth=h, kernel=k).fit(d[:, None]).score_samples([[x0]])[0])) for k in ['tophat', 'epanechnikov']}
A['47A.1'] = {'tophat': round(float(top), 4), 'epanechnikov': round(float(epa), 4), 'sklearn': {k: round(v, 4) for k, v in sk.items()}}
fig, ax = small(); g = np.linspace(-3, 13, 801)
for k, c, ls in [('tophat', GREY, '-'), ('epanechnikov', GREEN, '--')]: ax.plot(g, kde_curve(d[:, None], h, kernel=k, grid=g), color=c, ls=ls, lw=1.4, label=k)
ax.plot(d, np.zeros(4), 'o', color=BLUE, ms=4, clip_on=False, zorder=5); ax.axvline(x0, color=PUR, lw=0.9, ls=':')
ax.set_xlabel('$x$'); ax.set_ylabel('$\\hat f(x)$'); ax.set_ylim(0, 0.3); ax.set_yticks([0, 0.1, 0.2]); ax.set_xticks([0, 3, 6, 9, 12]); ax.legend(loc='upper right', fontsize=9, handlelength=1.6, borderaxespad=0.1)
save(fig, 'p_47A1.png')

# ---- 47A.2  Scott and Silverman factors
n = 100; ps = np.arange(1, 11); sc = n ** (-1 / (ps + 4.0)); si = (n * (ps + 2) / 4) ** (-1 / (ps + 4.0))
A['47A.2'] = {'scott': [round(float(v), 4) for v in sc[:3]], 'silverman': [round(float(v), 4) for v in si[:3]]}
fig, ax = small(); ax.plot(ps, sc, '-o', color=PUR, ms=3, lw=1.3, label='Scott'); ax.plot(ps, si, '--s', color=GREEN, ms=3, lw=1.3, label='Silverman')
ax.set_xlabel('number of features $p$'); ax.set_ylabel('factor, $n$ = 100'); ax.legend(loc='lower right', fontsize=9, handlelength=1.6, borderaxespad=0.1); ax.set_xticks([1, 2, 4, 6, 8, 10]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
save(fig, 'p_47A2.png')

# ---- 47A.3  leave-one-out for the four waits
t = np.array([52., 56., 80., 84.]); h = 4.0
phi = lambda v: np.exp(-v ** 2 / 2) / np.sqrt(2 * np.pi)
loo = [float(np.log(sum(phi((t[i] - t[j]) / h) for j in range(4) if j != i) / (3 * h))) for i in range(4)]
tr = [float(np.log(sum(phi((t[i] - t[j]) / h) for j in range(4)) / (4 * h))) for i in range(4)]
A['47A.3'] = {'loo_each': [round(v, 4) for v in loo], 'loo_mean': round(float(np.mean(loo)), 4), 'train_mean': round(float(np.mean(tr)), 4),
              'f_minus1': round(float(np.exp(loo[0])), 5)}
fig, ax = small(); g = np.linspace(38, 98, 601)
ax.plot(g, norm.pdf(g, 56, h) / 3 + norm.pdf(g, 80, h) / 3 + norm.pdf(g, 84, h) / 3, color=GREEN, lw=1.5)
ax.plot(g, norm.pdf(g, 52, h) / 4 * 0 + norm.pdf(g, 52, h) / 3, color=RED, lw=1.0, ls=':')
ax.plot(t, np.zeros(4), 'o', color=BLUE, ms=4, clip_on=False, zorder=5); ax.plot(52, np.exp(loo[0]), 'o', color=PUR, ms=4.5, zorder=6)
ax.set_xlabel('waiting time (min)'); ax.set_ylabel('density'); ax.set_ylim(0, 0.07); ax.set_yticks([0, 0.03, 0.06]); ax.set_xticks([40, 60, 80])
save(fig, 'p_47A3.png')

# ---- 47B.1  leave-one-out from the full sum
hh = np.arange(1.0, 10.01, 0.25); n = len(X); loo_cv = []
for hv in hh:
    c0 = 1 / (hv * np.sqrt(2 * np.pi)); f = np.exp(KernelDensity(bandwidth=hv).fit(X).score_samples(X))
    loo_cv.append(float(np.mean(np.log((n * f - c0) / (n - 1)))))
gs = GridSearchCV(KernelDensity(), {'bandwidth': hh}, cv=KFold(10, shuffle=True, random_state=0)).fit(X)
from sklearn.model_selection import LeaveOneOut
gl = GridSearchCV(KernelDensity(), {'bandwidth': hh}, cv=LeaveOneOut()).fit(X)
A['47B.1'] = {'h_best_identity': float(hh[int(np.argmax(loo_cv))]), 'cv_best_identity': round(max(loo_cv), 4), 'h_best_gridsearch_loo': float(gl.best_params_['bandwidth']),
              'cv_best_gridsearch_loo': round(float(gl.best_score_), 4), 'h_best_10fold': float(gs.best_params_['bandwidth'])}
fig, ax = small(); ax.plot(hh, loo_cv, color=GREEN, lw=1.5); ib = int(np.argmax(loo_cv)); ax.plot(hh[ib], loo_cv[ib], 'o', color=PUR, ms=5)
ax.set_xlabel('bandwidth $h$ (min)'); ax.set_ylabel('LOO mean $\\ln\\hat f$'); ax.set_ylim(-4.2, -3.7); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
save(fig, 'p_47B1.png')

# ---- 47B.2  raw against standardised iris petals
Xr = load_iris().data[:, 2:4]; sdv = Xr.std(0); Zs = (Xr - Xr.mean(0)) / sdv
def cvll(D, grid, seed=0):
    outer = KFold(5, shuffle=True, random_state=seed); out = []; hs_ = []
    for tr_, te_ in outer.split(D):
        g_ = GridSearchCV(KernelDensity(), {'bandwidth': grid}, cv=KFold(5, shuffle=True, random_state=1)).fit(D[tr_])
        hs_.append(float(g_.best_params_['bandwidth'])); out.append(g_.score(D[te_]) / len(te_))
    return float(np.mean(out)), hs_
llz, hz = cvll(Zs, np.arange(0.03, 0.50, 0.01)); llr, hr = cvll(Xr, np.arange(0.03, 1.0, 0.01))
jac = float(np.log(sdv).sum())
A['47B.2'] = {'scaled_ll': round(llz, 3), 'raw_ll': round(llr, 3), 'raw_ll_in_scaled_units': round(llr + jac, 3), 'sum_ln_sigma': round(jac, 3),
              'h_scaled': hz, 'h_raw': hr, 'sd_raw': [round(float(v), 3) for v in sdv]}
fig, ax = small(); b = ax.bar(['scaled', 'raw'], [llz, llr + jac], color=[GREEN, OR], width=0.5)
for r_, v in zip(b, [llz, llr + jac]): ax.text(r_.get_x() + r_.get_width() / 2, v + 0.04, '%.2f' % v, ha='center', va='bottom', fontsize=9, color='white', weight='bold')
ax.set_ylim(-1.0, 0); ax.set_ylabel('held-out mean $\\ln f$'); save(fig, 'p_47B2.png')

# ---- 47C.1  one bandwidth for tight and wide clusters
pl = StandardScaler().fit_transform(load_iris().data[:, 2:3])[:, 0]; g = np.linspace(-1.9, 1.9, 600)
fig, ax = small()
ax.plot(g, kde_curve(pl[:, None], 0.11, grid=g), color=GREEN, lw=1.3, label='$h$ = 0.11'); ax.plot(g, kde_curve(pl[:, None], 0.43, grid=g), color=PUR, lw=1.3, ls='--', label='$h$ = 0.43')
ax.plot(pl, np.full(len(pl), -0.1), '|', color=BLUE, ms=4, mew=0.4, clip_on=False); ax.set_xlabel('petal length (scaled)'); ax.set_ylabel('density')
ax.legend(loc='upper center', fontsize=9, handlelength=1.6, ncol=2, borderaxespad=0.1, columnspacing=0.8); ax.set_ylim(0, 1.5); ax.set_yticks([0, 0.5, 1.0])
save(fig, 'p_47C1.png')
json.dump(A, open('verify/problem_answers.json', 'w'), indent=1); print(json.dumps(A))
