# Figures 40.0-1 .. 40.3-1  (run: python3 fig/figs1.py)
from common import *
NUM = {}
X1, g1 = heights_1d()
gm1 = GM(2, tol=1e-8, max_iter=1000, random_state=0).fit(X1[:, None])
o = np.argsort(gm1.means_.ravel()); w = gm1.weights_[o]; m = gm1.means_.ravel()[o]; s = np.sqrt(gm1.covariances_.ravel()[o])
NUM['heights_fit'] = {'weights': w.round(4).tolist(), 'means': m.round(3).tolist(), 'sd': s.round(3).tolist(), 'n_iter': int(gm1.n_iter_)}
xg = np.linspace(135, 200, 600)
npdf = lambda x, mu, sd: np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))

# ---- 40.0-1: the chapter in one picture
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.3), dpi=240, gridspec_kw={'width_ratios': [1.15, 1]})
ax = axs[0]
ax.hist(X1, bins=np.arange(138, 200, 3), density=True, color=LGREY, ec='white', lw=0.6)
ax.plot(xg, w[0] * npdf(xg, m[0], s[0]), color=BLUE, lw=1.4, ls='--'); ax.plot(xg, w[1] * npdf(xg, m[1], s[1]), color=OR, lw=1.4, ls='--')
ax.plot(xg, w[0] * npdf(xg, m[0], s[0]) + w[1] * npdf(xg, m[1], s[1]), color=GREEN, lw=2.0)
ax.plot([], [], color=BLUE, lw=1.4, ls='--', label='group 1'); ax.plot([], [], color=OR, lw=1.4, ls='--', label='group 2'); ax.plot([], [], color=GREEN, lw=2.0, label='sum')
ax.legend(loc='upper right', fontsize=8.5, handlelength=1.8, borderaxespad=0.1, labelspacing=0.25)
ax.set_xlim(138, 200); ax.set_ylim(0, 0.058)
ax.set_xlabel('height (cm)'); ax.set_ylabel('density'); ax.set_yticks([0, 0.02, 0.04])
ax = axs[1]
p1 = w[0] * npdf(xg, m[0], s[0]); p2 = w[1] * npdf(xg, m[1], s[1]); r1 = p1 / (p1 + p2)
ax.fill_between(xg, 0, r1, color=BLUE, alpha=0.25, lw=0); ax.fill_between(xg, r1, 1, color=OR, alpha=0.25, lw=0)
ax.plot(xg, r1, color=GREEN, lw=2.0)
ax.text(147, 0.2, 'group 1', color=BLUE, fontsize=8.5); ax.text(193, 0.8, 'group 2', color=OR, fontsize=8.5, ha='right')
x50 = xg[np.argmin(abs(r1 - 0.5))]; ax.plot([x50], [0.5], 'o', color=GREEN, ms=4.5, mec='white', zorder=5); ax.text(x50 + 2.5, 0.52, f'{x50:.0f} cm: 50 : 50', fontsize=8.5, color=NAVY)
ax.set_xlim(138, 200); ax.set_ylim(0, 1); ax.set_xlabel('height (cm)'); ax.set_ylabel('responsibility of group 1')
NUM['x50'] = float(x50)
fig.tight_layout(w_pad=1.0); sv(fig, 'f400.png', 470)

# ---- 40.1-1: the six measurements
xs = np.array([154, 160, 164, 170, 176, 184.])
fig, ax = plt.subplots(figsize=(3.9, 1.3), dpi=240)
ax.spines['bottom'].set_position(('data', 0))
ax.scatter(xs, np.zeros(6), s=48, color=BLUE, edgecolor='white', lw=0.8, zorder=5)
for k, v in enumerate(xs): ax.text(v, 0.16, f'{int(v)}', ha='center', fontsize=8.5, color=NAVY)
ax.text(169, 0.78, 'which group?', fontsize=9, color=PUR, ha='center')
ax.set_xlim(146, 192); ax.set_ylim(-0.45, 1.0); ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_xlabel('height (cm)')
sv(fig, 'f411.png', 330)
dump('fig_numbers1.json', NUM); dump('fig_textsize1.json', CHK)
