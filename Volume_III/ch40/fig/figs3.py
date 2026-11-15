# Figures 40.4-1 and 40.5-1  (run: python3 fig/figs3.py)
from common import *
NUM = {}
xs = np.array([154, 160, 164, 170, 176, 184.]); pi = np.array([.5, .5]); var = 64.0; mu1 = 158.0
npdf = lambda x, mu, v: np.exp(-0.5 * (x - mu) ** 2 / v) / np.sqrt(2 * np.pi * v)


def comp(mu2):                       # pi_k N(x_i | mu_k, var) for the six points, shape (6, 2)
    return np.c_[pi[0] * npdf(xs, mu1, var), pi[1] * npdf(xs, mu2, var)]


ell_ = lambda mu2: np.log(comp(mu2).sum(1)).sum()
def bound(gam, mu2):                 # Eq. 40.4-1 with q = gam
    c = comp(mu2); return float(np.sum(gam * (np.log(c) - np.log(gam))))
def estep(mu2):
    c = comp(mu2); return c / c.sum(1, keepdims=True)

# ---- 40.4-1: the lower bound
mus = np.linspace(170.5, 181, 400)
L = np.array([ell_(m) for m in mus])
fig, ax = plt.subplots(figsize=(4.4, 2.5), dpi=240)
ax.plot(mus, L, color=NAVY, lw=2.0, label=r'$\ell(\theta)$, the log-likelihood')
cur = 178.0; cols = [BLUE, OR]; seq = [cur]; NUM['bound_steps'] = []
for it in range(2):
    gam = estep(cur); Fcur = np.array([bound(gam, m) for m in mus])
    new = float(gam[:, 1] @ xs / gam[:, 1].sum())            # M-step for mu2 (Eq. 40.4-5)
    ax.plot(mus, Fcur, color=cols[it], lw=1.5, ls='--', label=['bound built at the start', 'bound built after step 1'][it])
    ax.plot([cur], [ell_(cur)], 'o', color=cols[it], ms=5.5, mec='white', zorder=6)
    ax.plot([new], [bound(gam, new)], 's', color=cols[it], ms=5, mfc='white', zorder=6)
    ax.plot([new, new], [bound(gam, new), ell_(new)], color=cols[it], lw=0.8, ls=':')
    NUM['bound_steps'].append({'from': cur, 'to': new, 'ell_from': ell_(cur), 'bound_at_to': bound(gam, new), 'ell_to': ell_(new)})
    cur = new
ax.set_xlim(170.5, 181); ax.set_xlabel(r'$\mu_2$ (cm), the other parameters held fixed'); ax.set_ylabel('log-likelihood')
ax.set_ylim(-22.58, -22.375)
ax.legend(loc='upper right', fontsize=8.5, handlelength=1.8, borderaxespad=0.2, labelspacing=0.25)
grid(ax); sv(fig, 'f441.png', 400)
NUM['ell_max_mu2'] = float(mus[np.argmax(L)])

# ---- 40.5-1: soft assignment hardens as sigma shrinks
c1, c2 = 160.0, 178.0; xx = np.linspace(145, 193, 400)
fig, axs = plt.subplots(1, 4, figsize=(5.4, 1.7), dpi=240, sharey=True)
NUM['hard'] = {}
for ax, sd in zip(axs, [12, 6, 3, 1]):
    r = 1 / (1 + np.exp(-((xx - c2) ** 2 - (xx - c1) ** 2) / (2 * sd ** 2)))
    ax.fill_between(xx, 0, r, color=BLUE, alpha=0.22, lw=0); ax.fill_between(xx, r, 1, color=OR, alpha=0.22, lw=0)
    ax.plot(xx, r, color=GREEN, lw=1.8); ax.axvline((c1 + c2) / 2, color=GREY, lw=0.8, ls=':')
    ax.set_title(f'$\\sigma$ = {sd} cm', fontsize=9, color=NAVY); ax.set_xticks([150, 170, 190]); ax.set_xlim(145, 193); ax.set_ylim(0, 1)
    ax.set_xlabel('height', fontsize=9); ax.set_yticks([0, 0.5, 1])
axs[0].set_ylabel('responsibility of 1')
fig.tight_layout(w_pad=0.3); sv(fig, 'f451.png', 470)
dump('fig_numbers3.json', NUM); dump('fig_textsize3.json', CHK)
