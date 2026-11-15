# Problem figures of Chapter 40. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
from common import *
from gmm_scratch import MyGMM
from scipy.stats import norm
from sklearn.datasets import make_blobs, load_wine
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture as BGM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score as ari
plt.rcParams.update({'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.labelsize': 9.5, 'legend.fontsize': 9})
A = {}; PCHK = {}


def small(w=2.4, h=1.7): return plt.subplots(figsize=(w, h), dpi=240)


def psv(fig, name, wpx):
    fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()); save(fig, name)
    from PIL import Image
    with Image.open(name) as im: w, h = im.size
    pw = min(1.55 * wpx, 235 * w / h); PCHK[name] = round(fs * pw / 96 / (w / 240), 1)


# 40A.1 one E-step in one dimension
pi = np.array([0.4, 0.6]); mu = np.array([0., 3.]); sd = np.array([1., 1.])
def resp(x): d = pi * norm.pdf(x, mu, sd); return d / d.sum()
A['40A.1'] = {'x=2': resp(2.0).round(4).tolist(), 'x=1': resp(1.0).round(4).tolist(), 'N(2|0,1)': round(float(norm.pdf(2, 0, 1)), 5), 'N(2|3,1)': round(float(norm.pdf(2, 3, 1)), 5)}
fig, ax = small(2.9, 1.3); xx = np.linspace(-3, 6, 300)
ax.plot(xx, pi[0] * norm.pdf(xx, 0, 1), color=BLUE, lw=1.4); ax.plot(xx, pi[1] * norm.pdf(xx, 3, 1), color=OR, lw=1.4)
ax.axvline(1, color=PUR, lw=1.0, ls='--'); ax.text(0.9, 0.31, '$x$ = 1', color=PUR, fontsize=9, ha='right'); ax.axvline(2, color=PUR, lw=1.0, ls='--'); ax.text(2.1, 0.31, '$x$ = 2', color=PUR, fontsize=9, ha='left')
ax.set_xlabel('$x$'); ax.set_ylabel('$\\pi_k\\,\\mathcal{N}$'); ax.set_ylim(0, 0.34); ax.set_yticks([0, 0.1, 0.2, 0.3]); ax.set_xlim(-3, 6); psv(fig, 'p_40A1.png', 165)

# 40A.2 one M-step
x2 = np.array([1., 2., 6., 7.]); g1 = np.array([.9, .8, .2, .1]); G = np.c_[g1, 1 - g1]
Nk = G.sum(0); mu2 = G.T @ x2 / Nk; v2 = np.array([(G[:, k] * (x2 - mu2[k]) ** 2).sum() / Nk[k] for k in range(2)])
A['40A.2'] = {'N_k': Nk.tolist(), 'pi': (Nk / 4).tolist(), 'mu': mu2.round(4).tolist(), 'var': v2.round(4).tolist()}
fig, ax = small(2.9, 1.25)
for xv, gv in zip(x2, g1):
    ax.add_patch(plt.Circle((xv, 0.62), 0.5, fc=OR, ec='none', zorder=4)); ax.add_patch(matplotlib.patches.Wedge((xv, 0.62), 0.5, 90, 90 + 360 * gv, fc=BLUE, ec='none', zorder=5))
    ax.add_patch(plt.Circle((xv, 0.62), 0.5, fc='none', ec=NAVY, lw=0.8, zorder=6)); ax.text(xv, 1.3, f'{gv:.1f}', ha='center', fontsize=9, color=NAVY)
ax.set_xlim(-0.3, 8.3); ax.set_ylim(-0.05, 1.8); ax.set_aspect('equal'); ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_xticks(range(0, 9)); ax.set_xlabel('$x$')
psv(fig, 'p_40A2.png', 165)

# 40A.3 BIC and AIC
n = 500; ll = {'full': -3010., 'diag': -3040., 'spherical': -3105.}; K, p = 3, 4
q = {'full': (K - 1) + K * p + K * p * (p + 1) // 2, 'diag': (K - 1) + K * p + K * p, 'spherical': (K - 1) + K * p + K}
bic = {c: -2 * ll[c] + q[c] * np.log(n) for c in ll}; aic = {c: -2 * ll[c] + 2 * q[c] for c in ll}
A['40A.3'] = {'q': q, 'bic': {c: round(float(v), 1) for c, v in bic.items()}, 'aic': {c: round(float(v), 1) for c, v in aic.items()}, 'bic_best': min(bic, key=bic.get), 'aic_best': min(aic, key=aic.get)}
fig, ax = small(2.9, 1.3); xs_ = np.arange(3)
ax.bar(xs_ - 0.2, [bic[c] for c in ll], 0.38, color=BLUE, label='BIC'); ax.bar(xs_ + 0.2, [aic[c] for c in ll], 0.38, color=OR, label='AIC')
ax.set_xticks(xs_); ax.set_xticklabels(list(ll)); ax.set_ylim(6000, 6480); ax.set_ylabel('criterion'); ax.legend(loc='upper left', fontsize=9, ncol=2, handlelength=1, columnspacing=0.8, borderaxespad=0.1); psv(fig, 'p_40A3.png', 165)

# 40A.4 hardening
xx = np.linspace(-2, 6, 300); fig, ax = small(2.9, 1.3); res = {}
for sdv, ls_, col in ((2, '-', BLUE), (1, '--', PUR), (0.5, ':', RED)):
    ax.plot(xx, 1 / (1 + np.exp(-((xx - 4) ** 2 - xx ** 2) / (2 * sdv ** 2))), color=col, lw=1.4, ls=ls_, label=f'$\\sigma$ = {sdv:g}')
    res[str(sdv)] = float(1 / (1 + np.exp(-((1 - 4) ** 2 - 1 ** 2) / (2 * sdv ** 2))))
ax.axvline(1, color=GREY, lw=0.8, ls=':'); ax.set_xlabel('$x$'); ax.set_ylabel('$\\gamma_1$'); ax.set_ylim(0, 1.05); ax.set_xlim(-2, 6); ax.set_yticks([0, 0.5, 1])
ax.legend(loc='lower left', fontsize=9, handlelength=1.6, borderaxespad=0.1, labelspacing=0.2); A['40A.4'] = {k: round(v, 7) for k, v in res.items()}; psv(fig, 'p_40A4.png', 165)

# 40B.1 BIC method for MyGMM and a convergence trace
X, g = people_2d(); rng = np.random.default_rng(1); m0 = X[rng.choice(len(X), 3, replace=False)]; w0 = np.full(3, 1 / 3); S0 = np.array([np.cov(X.T)] * 3)
mine = MyGMM(w0, m0, S0.copy()).fit(X); ref = GM(3, weights_init=w0, means_init=m0, precisions_init=np.linalg.inv(S0)).fit(X)
q_ = 2 + 6 + 9; bic_mine = -2 * len(X) * mine.score_samples(X).mean() + q_ * np.log(len(X))
A['40B.1'] = {'bic_mine': round(float(bic_mine), 4), 'bic_sklearn': round(float(ref.bic(X)), 4), 'n_params': q_}
trace = [MyGMM(w0, m0, S0.copy(), tol=0, max_iter=t).fit(X).score_samples(X).mean() for t in range(1, 26)]
fig, ax = small(2.9, 1.3); ax.plot(range(1, 26), trace, color=GREEN, lw=1.5); ax.set_xlabel('EM step'); ax.set_ylabel('mean ln $p$'); ax.set_xticks([1, 10, 20]); grid(ax); psv(fig, 'p_40B1.png', 165)
A['40B.1']['trace_first_last'] = [round(float(trace[0]), 4), round(float(trace[-1]), 4)]

# 40B.2 anisotropic blobs: k-means against a mixture
Xb, yb = make_blobs(n_samples=500, centers=3, random_state=170); Xb = Xb @ np.array([[0.60834549, -0.63667341], [-0.40887718, 0.85253229]])
kmb = KMeans(3, n_init=10, random_state=0).fit(Xb); gmb = GM(3, n_init=5, random_state=0).fit(Xb)
A['40B.2'] = {'ari_kmeans': round(float(ari(yb, kmb.labels_)), 3), 'ari_gmm': round(float(ari(yb, gmb.predict(Xb))), 3)}
fig, ax = small(2.4, 1.4)
for k, (c_, mk_) in enumerate(zip(GCOL, GMK)): ax.scatter(Xb[yb == k, 0], Xb[yb == k, 1], s=6, color=c_, marker=mk_, lw=0)
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); psv(fig, 'p_40B2.png', 135)

# 40B.3 Dirichlet prior and the number of active components
Xc, _ = make_blobs(n_samples=300, centers=2, random_state=3); res = {}
fig, ax = small(2.9, 1.3)
for j, (gam0, c_) in enumerate(((0.01, GREEN), (100.0, PUR))):
    b = BGM(n_components=6, weight_concentration_prior_type='dirichlet_distribution', weight_concentration_prior=gam0, max_iter=500, tol=1e-6, random_state=0).fit(Xc)
    w_ = np.sort(b.weights_)[::-1]; res[str(gam0)] = {'active': int((w_ > 0.01).sum()), 'weights': w_.round(3).tolist()}
    ax.bar(np.arange(1, 7) + (j - 0.5) * 0.38, w_, 0.36, color=c_, label=f'$\\gamma_0$ = {gam0:g}')
ax.set_xlabel('component, sorted'); ax.set_ylabel('weight'); ax.set_xticks(range(1, 7)); ax.legend(loc='upper right', fontsize=9, handlelength=1, borderaxespad=0.1, labelspacing=0.2); A['40B.3'] = res; psv(fig, 'p_40B3.png', 165)

# 40C.1 the likelihood has no maximum
xs6 = np.array([154, 160, 164, 170, 176, 184.]); sds = np.logspace(-3, np.log10(4), 80)
def ll6(sd):
    pp = np.array([.5, .5]); mm = np.array([160., 170.]); vv = np.array([100., sd ** 2]); dd = pp * norm.pdf(xs6[:, None], mm, np.sqrt(vv)); return np.log(dd.sum(1)).sum()
fig, ax = small(2.9, 1.3); ax.semilogx(sds, [ll6(s) for s in sds], color=RED, lw=1.5); ax.set_xlabel('$\\sigma_2$ (cm)'); ax.set_ylabel('$\\ell$'); ax.invert_xaxis(); grid(ax)
ax.axhline(-21.868, color=GREEN, lw=1.0, ls='--'); ax.text(3, -21.2, 'best proper fit', color=GREEN, fontsize=9, va='bottom'); psv(fig, 'p_40C1.png', 165)

# 40C.2 BIC and the number of cultivars (numbers from the wine case study)
import json as _j
NB = _j.load(open('verify/fig_numbers5.json'))['wine_bic']
fig, ax = small(2.9, 1.3); ax.plot(range(1, 7), NB['diag'], color=OR, marker='^', ms=4, lw=1.4, ls='-.'); ax.axvline(3, color=GREY, lw=0.9, ls=':'); ax.text(3.1, 6600, 'three\ncultivars', fontsize=9, color=GREY, va='center')
ax.set_xlabel('$K$ (diag)'); ax.set_ylabel('BIC'); ax.set_xticks(range(1, 7)); ax.set_ylim(5400, 6800); grid(ax); psv(fig, 'p_40C2.png', 165)

# 40C.3 label switching
Xw, yw = load_wine(return_X_y=True); Xw = StandardScaler().fit_transform(Xw); tabs = []; accs = []
fig, axs = plt.subplots(1, 2, figsize=(2.9, 1.5), dpi=240)
for ax, s in zip(axs, (0, 3)):
    m = GM(3, covariance_type='diag', init_params='random_from_data', random_state=s, tol=1e-5, max_iter=500).fit(Xw); lab = m.predict(Xw)
    T = np.array([[np.sum((yw == c) & (lab == k)) for k in range(3)] for c in range(3)]); tabs.append(T.tolist()); accs.append(float((lab == yw).mean()))
    ax.imshow(T, cmap='Blues', vmin=0, vmax=70)
    for i in range(3):
        for j in range(3): ax.text(j, i, str(T[i, j]), ha='center', va='center', fontsize=9, color=NAVY if T[i, j] < 40 else 'white')
    ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xlabel(f'component (seed {s})', fontsize=9)
axs[0].set_ylabel('cultivar'); axs[1].set_yticklabels([])
fig.tight_layout(w_pad=0.5); psv(fig, 'p_40C3.png', 165); A['40C.3'] = {'tables': tabs, 'raw_accuracy': [round(a, 3) for a in accs]}
json.dump(A, open('verify/problem_answers.json', 'w'), indent=1, default=float); json.dump(PCHK, open('verify/probfig_textsize.json', 'w'), indent=1)
