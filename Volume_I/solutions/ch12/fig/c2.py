# Problem 12C.2: evidence against cross-validation. Writes fig/s12c2.png
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.datasets import load_diabetes
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

d = load_diabetes(); names = list(d.feature_names)
Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=0.25, random_state=0)
rat = np.logspace(-6, 3, 55)
def cv(X, y, r): return -cross_val_score(Ridge(alpha=r), X, y, cv=5, scoring='neg_mean_squared_error').mean()
def study(X, y, tag):
    b = BayesianRidge().fit(X, y); rb = b.lambda_ / b.alpha_; c = np.array([cv(X, y, r) for r in rat]); k = int(c.argmin())
    print(f"{tag}: evidence ratio {rb:.4g} (CV mse there {cv(X, y, rb):.1f}); CV ratio on the grid {rat[k]:.3g} (CV mse {c[k]:.1f}); excess {100 * (cv(X, y, rb) / c[k] - 1):.1f} per cent; noise sd {b.alpha_ ** -0.5:.1f}")
    return rb, c, b
print("== the diabetes data as they are")
rb0, c0, b0 = study(Xtr, ytr, "diabetes")
print("relative range of the CV error over ratios 0.003 to 0.3: %.3f" % (max(cv(Xtr, ytr, r) for r in (0.003, 0.03, 0.3)) / min(cv(Xtr, ytr, r) for r in (0.003, 0.03, 0.3)) - 1))
print("\n== the same data, with the columns bmi, bp and s5 multiplied by 0.01 (units 100 times larger)")
X1 = Xtr.copy()
for nm in ('bmi', 'bp', 's5'): X1[:, names.index(nm)] *= 0.01
rb1, c1, b1 = study(X1, ytr, "rescaled")
print("weights of bmi, bp, s5 in least squares: original", np.round(Ridge(alpha=1e-9).fit(Xtr, ytr).coef_[[2, 3, 8]], 0), " rescaled", np.round(Ridge(alpha=1e-9).fit(X1, ytr).coef_[[2, 3, 8]], 0))
Z = StandardScaler().fit_transform(X1); rb2, c2, b2 = study(Z, ytr, "rescaled, then standardised")
Z0 = StandardScaler().fit_transform(Xtr); study(Z0, ytr, "original, standardised")
t_ev = ((Ridge(alpha=rb1).fit(X1, ytr).predict(Xte * np.where(np.isin(names, ['bmi', 'bp', 's5']), 0.01, 1.0)) - yte) ** 2).mean()
t_cv = ((Ridge(alpha=rat[c1.argmin()]).fit(X1, ytr).predict(Xte * np.where(np.isin(names, ['bmi', 'bp', 's5']), 0.01, 1.0)) - yte) ** 2).mean()
print("test mse on the 111 held-out patients: evidence ratio %.0f, CV ratio %.0f" % (t_ev, t_cv))

fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.95), dpi=240, sharey=False)
for ax, c, rb, ttl in ((axs[0], c0, rb0, 'diabetes data'), (axs[1], c1, rb1, 'three columns rescaled')):
    ax.semilogx(rat, c, color=GREY, lw=1.4); ax.plot(rat[c.argmin()], c.min(), 'o', color=BLUE, ms=5, mec='white', zorder=5, label='cross-validation')
    ax.plot(rb, np.interp(np.log(rb), np.log(rat), c), 's', color=GREEN, ms=5, mec='white', zorder=5, label='evidence')
    ax.set_title(ttl, fontsize=8.5, color=GREY, pad=3); ax.set_xlabel('$\\lambda/\\alpha$', fontsize=9); ax.tick_params(labelsize=8.5); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
axs[0].set_ylim(2900, 3700); axs[1].set_ylim(2900, 6000); axs[0].set_ylabel('CV mean squared error', fontsize=8.5)
axs[0].legend(fontsize=8.5, loc='upper left', borderpad=0.2, handlelength=1.0, labelspacing=0.2); axs[0].set_xlim(1e-4, 1e2); axs[1].set_xlim(1e-6, 1e3)
fig.tight_layout(pad=0.4, w_pad=0.8)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s12c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s12c2.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))
