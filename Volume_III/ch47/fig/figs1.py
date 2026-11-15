# Figures 47.0-1 to 47.3-1 and the algorithm chart. Run: python3 fig/figs1.py
from common import *

# ---- 47.0-1  the whole idea
fig, ax = plt.subplots(figsize=(4.9, 1.45), dpi=240)
ax.hist(X[:, 0], bins=np.arange(40, 100.1, 4), density=True, color=LB, edgecolor=BLUE, lw=0.6, zorder=1)
ax.plot(XGRID, truth(XGRID), color=GREY, lw=1.2, ls='--', zorder=3)
ax.plot(XGRID, kde_curve(X, 2.5), color=GREEN, lw=2.2, zorder=4)
ax.plot(X[:, 0], np.full(len(X), -0.0016), '|', color=BLUE, ms=6, mew=0.6, clip_on=False, zorder=5)
ax.text(46.5, 0.049, 'histogram\n(bins of 4 min)', color=BLUE, fontsize=9, ha='left', va='center')
ax.text(86.5, 0.041, 'kernel density\nestimate, $h$ = 2.5', color=GREEN, fontsize=9, weight='bold', ha='left')
ax.text(60.5, 0.017, 'true density\n(dashed)', color=GREY, fontsize=9, ha='left')
ax.set_xlim(38, 102); ax.set_ylim(0, 0.058); ax.set_xlabel('waiting time (min); one tick per observation')
ax.set_ylabel('density (per min)'); ax.set_yticks([0, 0.02, 0.04])
save(fig, 'f470.png')

# ---- 47.2-1  histogram: bin width and origin
sub = X[:60, 0]
fig, axs = plt.subplots(1, 3, figsize=(4.9, 1.5), dpi=240, sharey=True)
specs = [(8, 40, 'width 8, origin 40'), (8, 44, 'width 8, origin 44'), (3, 40, 'width 3, origin 40')]
NUM['hist'] = {}
for ax, (w, o, t) in zip(axs, specs):
    edges = np.arange(o - w, 105, w)
    cnt, _ = np.histogram(sub, bins=edges)
    ax.bar(edges[:-1], cnt / (len(sub) * w), width=w, align='edge', color=LB, edgecolor=BLUE, lw=0.6)
    ax.plot(sub, np.full(len(sub), -0.0025), '|', color=BLUE, ms=4, mew=0.5, clip_on=False)
    ax.set_title(t, fontsize=8.5, color=NAVY); ax.set_xlim(38, 102); ax.set_xticks([40, 60, 80, 100])
    NUM['hist'][t] = {'max_density': round(float(cnt.max() / (len(sub) * w)), 4), 'n_bins_used': int((cnt > 0).sum())}
axs[0].set_ylabel('density (per min)'); axs[0].set_yticks([0, 0.02, 0.04])
fig.tight_layout(w_pad=0.5); save(fig, 'f472.png')

# ---- 47.3-1  bumps, and the kernels
toy = np.array([52., 56., 80., 84.]); h = 4.0
fig, axs = plt.subplots(1, 2, figsize=(4.9, 1.75), dpi=240, gridspec_kw={'width_ratios': [1.05, 1]})
ax = axs[0]; g = np.linspace(38, 98, 601)
for t in toy:
    ax.plot(g, norm.pdf(g, t, h) / 4, color=OR, lw=1.0)
ax.plot(g, kde_curve(toy[:, None], h, grid=g), color=GREEN, lw=2.0)
ax.plot(toy, np.zeros(4), 'o', color=BLUE, ms=4.5, clip_on=False, zorder=5)
f58 = float(kde_curve(toy[:, None], h, grid=np.array([58.0]))[0]); NUM['f58'] = f58
ax.plot([58, 58], [0, f58], color=PUR, lw=1.0, ls=':'); ax.plot(58, f58, 'o', color=PUR, ms=4.5, zorder=6)
ax.annotate('$\\hat f(58)$ = %.4f' % f58, xy=(58, f58), xytext=(58, 0.0545), color=PUR, fontsize=8.5, ha='center', va='center', arrowprops=dict(arrowstyle='-', color=PUR, lw=0.7))
ax.text(38, 0.0725, 'orange: four bumps, each of area 1/4', color=OR, fontsize=8.5, ha='left', va='center')
ax.text(38, 0.0655, 'green: their sum', color=GREEN, fontsize=8.5, ha='left', va='center')
ax.set_xlabel('waiting time (min)'); ax.set_ylabel('density (per min)'); ax.set_ylim(0, 0.077); ax.set_yticks([0, 0.02, 0.04]); ax.set_xlim(36, 100)
ax = axs[1]; u = np.linspace(-2.4, 4.3, 671)
names = [('gaussian', GREEN, '-'), ('tophat', GREY, '-'), ('epanechnikov', BLUE, '-'), ('exponential', RED, '--'), ('linear', OR, '-.'), ('cosine', PUR, ':')]
for k, c, ls in names:
    ax.plot(u, kde_curve(np.zeros((1, 1)), 1.0, kernel=k, grid=u), color=c, ls=ls, lw=1.4, label=k)
ax.set_xlabel('$u = (x - x_i)/h$'); ax.set_ylabel('$K(u)$'); ax.set_xlim(-2.4, 4.3); ax.set_ylim(0, 1.85); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.legend(fontsize=8.5, loc='upper right', ncol=2, handlelength=1.4, labelspacing=0.15, columnspacing=0.8, borderaxespad=0.1)
fig.tight_layout(w_pad=1.0); save(fig, 'f473.png')

dump('fig_numbers1')
print(json.dumps(NUM))
