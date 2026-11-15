# Problem 11C.3: the grid of ElasticNetCV for small l1_ratio. Writes fig/s11c3.png
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split

d = load_diabetes(); Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=0.25, random_state=0)
Xc, yc = Xtr - Xtr.mean(0), ytr - ytr.mean(); n = len(ytr)
amax = np.abs(Xc.T @ yc).max() / n; v = (Xc ** 2).mean(0)
print("alpha_max (lasso) = %.4f ; mean squares of the columns: %.4f to %.4f" % (amax, v.min(), v.max()))
print("\nrho   grid start = amax/rho   grid end   L2 strength at end   shrink v/(v+L2)   alpha_ chosen   last point?   CV mse   test R2")
curves = {}
for rho in (0.5, 0.1, 0.01):
    m = ElasticNetCV(l1_ratio=rho, cv=5, random_state=0).fit(Xtr, ytr)
    end = m.alphas_[-1]; l2 = end * (1 - rho); mean = m.mse_path_.mean(1); curves[rho] = (m.alphas_, mean, m.alpha_)
    print(f"{rho:4.2f} {m.alphas_[0]:12.3f} {end:20.4f} {l2:14.4f} {v.mean() / (v.mean() + l2):17.4f} {m.alpha_:15.4f} {str(bool(m.alpha_ == end)):>12s} {mean.min():9.1f} {m.score(Xte, yte):8.3f}")
print("\nwith eps = 1e-6 (a grid 10^6 long):")
ext = {}
for rho in (0.5, 0.1, 0.01):
    m = ElasticNetCV(l1_ratio=rho, eps=1e-6, cv=5, random_state=0).fit(Xtr, ytr); mean = m.mse_path_.mean(1); ext[rho] = (m.alphas_, mean, m.alpha_)
    print(f"rho {rho:4.2f}: grid {m.alphas_[0]:.3f} ... {m.alphas_[-1]:.2e}, alpha_ = {m.alpha_:.2e} (last point: {bool(m.alpha_ == m.alphas_[-1])}), CV mse {mean.min():.1f}, test R2 {m.score(Xte, yte):.3f}, non-zero {int((m.coef_ != 0).sum())}")
try: ElasticNetCV(l1_ratio=0, cv=5).fit(Xtr, ytr)
except ValueError as e: print("\nl1_ratio=0:", type(e).__name__ + ":", str(e)[:75])
grid = np.logspace(-4, 0, 30)
m0 = ElasticNetCV(l1_ratio=0, alphas=grid, cv=5).fit(Xtr, ytr); print("l1_ratio=0 with alphas=logspace(-4, 0, 30): alpha_ = %.2e, CV mse %.1f, test R2 %.3f" % (m0.alpha_, m0.mse_path_.mean(1).min(), m0.score(Xte, yte)))

fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.95), dpi=240, sharey=True)
cols = {0.5: BLUE, 0.1: OR, 0.01: PUR}
for ax, dct, ttl in zip(axs, (curves, ext), ('default grid (eps = 1e-3)', 'long grid (eps = 1e-6)')):
    for rho, (al, mse, ach) in dct.items():
        ax.plot(al, mse, color=cols[rho], lw=1.4, label=f'$\\rho$ = {rho}'); k = np.argmin(mse); ax.plot(al[k], mse[k], 'o', color=cols[rho], ms=4.5, mec='white', zorder=5)
    ax.set_xscale('log'); ax.set_xlabel('$\\alpha$', fontsize=9); ax.set_title(ttl, fontsize=8.5, color=GREY, pad=3); ax.tick_params(labelsize=8.5); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
axs[0].set_ylabel('CV mean squared error', fontsize=8.5); axs[0].legend(fontsize=8.5, loc='lower right', borderpad=0.2, handlelength=1.4, labelspacing=0.2)
axs[0].set_ylim(2800, 7000); fig.tight_layout(pad=0.4, w_pad=0.6)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s11c3.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s11c3.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))
