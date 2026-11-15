# Problem 35C.1: a feature with a huge spread and no relation to the target. Writes fig/s35c1.png.
import sys, os, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.decomposition import PCA
np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(7); n = 70; z = rng.standard_normal((n, 2))       # the data of the problem figure
X = np.c_[40 * z[:, 0], 1.0 * z[:, 1]]; y = X[:, 1] + 0.3 * rng.standard_normal(n); X -= X.mean(0); y -= y.mean()
sd = X.std(0, ddof=1); print("sd of x1, x2: %s, sd of y %.3f" % (sd, y.std(ddof=1)))
print("correlation of x1 with y: %+.4f   of x2 with y: %+.4f" % tuple(np.corrcoef(X[:, j], y)[0, 1] for j in (0, 1)))
g = X.T @ y; print("X^T y = %s  (divided by n - 1: %s)" % (g, g / (n - 1)))
print("ratio |c1| / |c2| = %.3f" % abs(g[0] / g[1]))
def rep(name, w, M):
    w = w / np.linalg.norm(w); t = M @ w; print("%-28s |w| = %s   corr(t, y) = %.3f   sd(t) = %.2f" % (name, np.abs(w), abs(np.corrcoef(t, y)[0, 1]), t.std(ddof=1)))
    return w
w_pca = rep("PCA, raw", PCA(1).fit(X).components_[0], X)
w_pls = rep("PLS, raw (Eq. 35.3-1)", g, X)
Z = X / sd; gz = Z.T @ y
w_plz = rep("PLS, standardised", gz, Z)
w_pcz = rep("PCA, standardised", PCA(1).fit(Z).components_[0], Z)
print("standardised X^T y = %s  (divided by n - 1 and sd(y): the correlations %s)" % (gz, gz / (n - 1) / y.std(ddof=1)))
print("correlation of x1 with y above which x1 gets the larger raw PLS weight: %.4f" % (abs(g[1]) / (n - 1) / sd[0] / y.std(ddof=1)))
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.0), dpi=240, gridspec_kw={'width_ratios': [1.25, 1]})
ax = axs[0]; ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=11, edgecolor='#666666', lw=0.25, zorder=2)
for w, col, lab, ls in ((w_pca, GREY, 'PCA', '--'), (w_pls, OR, 'PLS raw', '-'), (w_plz, GREEN, 'PLS standardised', '-')):
    if lab == 'PLS standardised':
        wd = np.array([w[0] * sd[0], w[1] * sd[1]]); wd /= np.linalg.norm(wd)     # the direction in the raw axes
    else: wd = w
    ax.plot([-wd[0] * 120, wd[0] * 120], [-wd[1] * 120, wd[1] * 120], color=col, lw=1.4, ls=ls, label=lab, zorder=3)
ax.set_xlim(-110, 110); ax.set_ylim(-3.4, 3.4); ax.set_xlabel('$x_1$  (sd 40)'); ax.set_ylabel('$x_2$  (sd 1)'); ax.legend(loc='upper left', fontsize=8.8, handlelength=1.4, borderaxespad=0.1, frameon=True, facecolor='white', edgecolor='none', framealpha=0.9)
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax = axs[1]; names = ['PCA', 'PLS\nraw', 'PLS\nstd.']; Ts = [X @ w_pca, X @ w_pls, Z @ w_plz]
vals = [abs(np.corrcoef(t, y)[0, 1]) for t in Ts]
ax.bar(range(3), vals, color=[GREY, OR, GREEN], width=0.6)
for k, v in enumerate(vals): ax.text(k, v + 0.02, '%.2f' % v, ha='center', fontsize=8.8, color=NAVY)
ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=8.8); ax.set_ylabel('|corr(score, $y$)|'); ax.set_ylim(0, 1.15); ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.subplots_adjust(wspace=0.38, left=0.09, right=0.99, top=0.97, bottom=0.2)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s35c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
w = Image.open('fig/s35c1.png').size[0] / 240; print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))
