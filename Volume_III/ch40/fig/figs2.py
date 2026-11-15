# Figures 40.2-1 .. 40.5-1  (run: python3 fig/figs2.py)
from common import *
from gmm_scratch import MyGMM
NUM = {}
X, g = people_2d()
xlim = (100, 205); ylim = (10, 112)

# ---- 40.2-1: EM in pictures
w0 = np.full(3, 1 / 3); m0 = np.array([[130., 70.], [160., 40.], [175., 95.]]); S0 = np.array([np.eye(2) * 150.] * 3)
steps = [0, 1, 10, 200]
fig, axs = plt.subplots(1, 4, figsize=(5.4, 1.85), dpi=240, sharex=True, sharey=True)
NUM['em_steps'] = {}
for ax, t in zip(axs, steps):
    mod = MyGMM(w0, m0, S0.copy(), tol=0, max_iter=max(t, 1))
    if t > 0: mod.fit(X)
    lse, R = mod._e_step(X)
    ax.scatter(X[:, 0], X[:, 1], s=7, c=blend(R), lw=0, zorder=3)
    for k in range(3): ell(ax, mod.mu_[k], mod.cov_[k], nstd=2, color=GREEN, lw=1.3)
    ax.scatter(mod.mu_[:, 0], mod.mu_[:, 1], s=22, marker='+', color=NAVY, lw=1.2, zorder=6)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_title('start' if t == 0 else (f'{t} step' + ('s' if t > 1 else '')), fontsize=9, color=NAVY)
    ax.set_xticks([120, 160, 200]); ax.set_xlabel('height', fontsize=9)
    NUM['em_steps'][str(t)] = {'mean_loglik': float(lse.mean()), 'weights': mod.pi_.round(3).tolist()}
axs[0].set_ylabel('weight'); axs[0].set_yticks([20, 60, 100])
fig.tight_layout(w_pad=0.3); sv(fig, 'f421.png', 470)

# ---- 40.3-1: the hidden label
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.0), dpi=240, sharex=True, sharey=True)
ax = axs[0]
for k in range(3): ax.scatter(X[g == k, 0], X[g == k, 1], s=9, color=GCOL[k], marker=GMK[k], lw=0, zorder=3)
ax.set_title('with the hidden label $z$', fontsize=9, color=NAVY)
ax = axs[1]; ax.scatter(X[:, 0], X[:, 1], s=9, color=GREY, lw=0, zorder=3); ax.set_title('what is observed', fontsize=9, color=NAVY)
for ax in axs: ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_xlabel('height (cm)', fontsize=9)
axs[0].set_ylabel('weight (kg)')
for k, nm in enumerate(('children', 'women', 'men')): axs[0].scatter([], [], s=14, color=GCOL[k], marker=GMK[k], label=nm)
axs[0].legend(loc='upper left', fontsize=8.5, handletextpad=0.2, borderaxespad=0.1, labelspacing=0.2)
fig.tight_layout(w_pad=0.6); sv(fig, 'f431.png', 400)
dump('fig_numbers2.json', NUM); dump('fig_textsize2.json', CHK)
