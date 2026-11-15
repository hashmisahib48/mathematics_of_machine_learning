# Figures 40.6-1, 40.12-1, 40.12-2  (run: python3 fig/figs4.py)
from common import *
NUM = {}
X, g = people_2d()
xlim = (100, 205); ylim = (10, 112)


def full_cov(m):
    """covariances as (K, p, p) for every covariance_type"""
    K, p = m.means_.shape
    if m.covariance_type == 'full': return m.covariances_
    if m.covariance_type == 'tied': return np.array([m.covariances_] * K)
    if m.covariance_type == 'diag': return np.array([np.diag(c) for c in m.covariances_])
    return np.array([np.eye(p) * c for c in m.covariances_])

# ---- 40.6-1: four covariance types
fig, axs = plt.subplots(1, 4, figsize=(5.4, 1.95), dpi=240, sharex=True, sharey=True)
NUM['types'] = {}
for ax, ct in zip(axs, ['full', 'tied', 'diag', 'spherical']):
    m = GM(3, covariance_type=ct, n_init=3, tol=1e-5, max_iter=500, random_state=0).fit(X)
    ax.scatter(X[:, 0], X[:, 1], s=5, color=GRID, lw=0, zorder=2)
    for k, C_ in enumerate(full_cov(m)): ell(ax, m.means_[k], C_, nstd=2, color=GREEN, lw=1.4)
    ax.scatter(m.means_[:, 0], m.means_[:, 1], s=14, marker='+', color=NAVY, lw=1.0, zorder=6)
    ax.set_title(ct, fontsize=9, color=NAVY); ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_xticks([120, 160, 200]); ax.set_xlabel('height', fontsize=9)
    ax.text(0.04, 0.96, f'{m._n_parameters()} parameters\nBIC {m.bic(X):.0f}', transform=ax.transAxes, fontsize=8.5, color=NAVY, va='top', ha='left', linespacing=1.1)
    NUM['types'][ct] = {'n_parameters': int(m._n_parameters()), 'bic': float(m.bic(X)), 'aic': float(m.aic(X)), 'loglik': float(m.score(X)), 'weights': np.sort(m.weights_).round(3).tolist()}
axs[0].set_ylabel('weight'); axs[0].set_yticks([20, 60, 100])
fig.tight_layout(w_pad=0.3); sv(fig, 'f461.png', 470)

# ---- 40.12-1: four starts, four endings
seeds = [0, 7, 14, 4]; fig, axs = plt.subplots(1, 4, figsize=(5.4, 1.95), dpi=240, sharex=True, sharey=True)
NUM['starts'] = {}
for ax, s in zip(axs, seeds):
    m = GM(3, init_params='random_from_data', random_state=s, tol=1e-7, max_iter=2000).fit(X)
    ax.scatter(X[:, 0], X[:, 1], s=5, color=GRID, lw=0, zorder=2)
    for k in range(3): ell(ax, m.means_[k], m.covariances_[k], nstd=2, color=GREEN, lw=1.4)
    ax.scatter(m.means_[:, 0], m.means_[:, 1], s=14, marker='+', color=NAVY, lw=1.0, zorder=6)
    ax.set_title(f'seed {s}', fontsize=9, color=NAVY); ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_xticks([120, 160, 200]); ax.set_xlabel('height', fontsize=9)
    ax.text(0.97, 0.04, f'{m.lower_bound_:.4f}\n{m.n_iter_} steps', transform=ax.transAxes, fontsize=8.5, color=NAVY, va='bottom', ha='right', linespacing=1.1)
    NUM['starts'][str(s)] = {'loglik': float(m.lower_bound_), 'n_iter': int(m.n_iter_), 'weights': np.sort(m.weights_).round(3).tolist()}
axs[0].set_ylabel('weight'); axs[0].set_yticks([20, 60, 100])
fig.tight_layout(w_pad=0.3); sv(fig, 'f4121.png', 470)
# how often does a random start find the best fit?
ll = []
for s in range(30):
    m = GM(3, init_params='random_from_data', random_state=s, tol=1e-7, max_iter=2000).fit(X); ll.append(m.lower_bound_)
ll = np.array(ll); NUM['starts_summary'] = {'n': 30, 'best': float(ll.max()), 'n_best': int((ll > ll.max() - 1e-3).sum()), 'worst': float(ll.min())}
km = [GM(3, init_params='kmeans', random_state=s, tol=1e-7, max_iter=2000).fit(X).lower_bound_ for s in range(30)]
NUM['starts_summary']['kmeans_n_best'] = int((np.array(km) > ll.max() - 1e-3).sum())

# ---- 40.12-2: choosing K and the covariance type; train and test
rng = np.random.default_rng(0); idx = rng.permutation(len(X)); itr, ite = idx[:350], idx[350:]
Ks = np.arange(1, 9); cts = ['full', 'tied', 'diag', 'spherical']
fig, axs = plt.subplots(1, 2, figsize=(5.4, 1.95), dpi=240, gridspec_kw={'width_ratios': [1.1, 1]})
ax = axs[0]; NUM['bic'] = {}
sty = {'full': (GREEN, '-', 'o'), 'tied': (PUR, '--', 's'), 'diag': (OR, '-.', '^'), 'spherical': (BLUE, ':', 'v')}
for ct in cts:
    b = [GM(k, covariance_type=ct, n_init=3, tol=1e-5, max_iter=500, random_state=0).fit(X).bic(X) for k in Ks]
    NUM['bic'][ct] = [round(float(v), 1) for v in b]
    c_, ls_, mk_ = sty[ct]; ax.plot(Ks, b, color=c_, ls=ls_, marker=mk_, ms=3.5, lw=1.3, label=ct)
    j = int(np.argmin(b)); ax.plot([Ks[j]], [b[j]], marker='*', color=c_, ms=11, mec='white', zorder=6)
ax.set_xlabel('number of components $K$'); ax.set_ylabel('BIC (lower is better)'); ax.set_ylim(7500, 8200); ax.set_xticks(Ks); grid(ax)
ax.legend(loc='upper right', fontsize=8.5, handlelength=2.0, borderaxespad=0.1, labelspacing=0.2)
ax = axs[1]; tr, te = [], []
for k in Ks:
    m = GM(k, covariance_type='full', n_init=3, tol=1e-5, max_iter=500, random_state=0).fit(X[itr]); tr.append(m.score(X[itr])); te.append(m.score(X[ite]))
NUM['trte'] = {'train': [round(float(v), 4) for v in tr], 'test': [round(float(v), 4) for v in te], 'best_test_K': int(Ks[int(np.argmax(te))])}
ax.plot(Ks, tr, color=BLUE, marker='o', ms=3.5, lw=1.4, label='training'); ax.plot(Ks, te, color=RED, marker='s', ms=3.5, lw=1.4, ls='--', label='test')
ax.set_xlabel('number of components $K$'); ax.set_ylabel('mean log-likelihood'); ax.set_xticks(Ks); ax.set_ylim(-8.3, -7.2); grid(ax)
ax.legend(loc='lower right', fontsize=8.5, handlelength=2.0, borderaxespad=0.1, labelspacing=0.2)
fig.tight_layout(w_pad=1.0); sv(fig, 'f4122.png', 470)
dump('fig_numbers4.json', NUM); dump('fig_textsize4.json', CHK)
