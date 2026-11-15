# Problem 40C.1: the unbounded likelihood and what reg_covar does. Writes fig/s40c1.png, prints the numbers.
import sys, os, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning); warnings.simplefilter('ignore', UserWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

x = np.array([154, 160, 164, 170, 176, 184.])[:, None]
def ll6(sd):                                         # component 2 on person 4, component 1: weight 1/2, mean 160, variance 100
    d = 0.5 * norm.pdf(x, [160., 170.], np.sqrt([100., sd ** 2])); return np.log(d.sum(1)).sum()
proper = GaussianMixture(2, tol=1e-8, max_iter=500, weights_init=[.5, .5], means_init=[[158.], [178.]],
                         precisions_init=[[[1 / 64]], [[1 / 64]]]).fit(x).score(x) * 6
print("proper fit of Section 40.11: ln L = %.3f; variance of the six heights %.1f cm^2" % (proper, x.var()))
for sd in (4, 1, 0.1, 0.01, 0.001): print("  sigma_2 = %-6g ln L = %.3f" % (sd, ll6(sd)))
print("  each factor 10 in sigma_2 adds ln 10 = %.4f: %.4f" % (np.log(10), ll6(0.001) - ll6(0.01)))
star = np.exp(brentq(lambda t: ll6(np.exp(t)) - proper, np.log(1e-4), np.log(1)))
print("the slice crosses the proper fit at sigma_2 = %.4f cm" % star)
# the density of person 4 from component 2 alone, 0.5 / (sigma sqrt(2 pi))
for sd in (0.1, 0.01, 0.001): print("  sigma_2 = %-6g density of person 4 from component 2: %.1f" % (sd, 0.5 / (sd * np.sqrt(2 * np.pi))))

def spike(reg):                                      # EM started with component 2 on person 4 and a tiny variance
    g = GaussianMixture(2, means_init=[[160.], [170.]], weights_init=[.5, .5], precisions_init=[[[1 / 100.]], [[1 / 1e-4]]],
                        reg_covar=reg, tol=1e-10, max_iter=1000).fit(x)
    return g, g.score(x) * 6
regs = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 0.3, 1, 4]
print("EM from the collapsed start, for several reg_covar (cm^2):")
res = []
for r in regs:
    g, l = spike(r); res.append(l)
    print("  reg_covar %-7g -> sd of component 2 %.4f, weight %.3f, ln L %.3f %s" % (r, np.sqrt(g.covariances_.ravel().min()), g.weights_[np.argmin(g.covariances_.ravel())], l, "(above the proper fit)" if l > proper else "(below)"))
thr = brentq(lambda t: spike(np.exp(t))[1] - proper, np.log(0.1), np.log(1))
print("the spike falls below the proper fit for reg_covar above %.2f cm^2 (floor sd %.2f cm)" % (np.exp(thr), np.sqrt(np.exp(thr))))

fig, axs = plt.subplots(1, 2, figsize=(5.4, 1.85), dpi=240)
ax = axs[0]; sds = np.logspace(-3, np.log10(4), 120)
ax.semilogx(sds, [ll6(s) for s in sds], color=RED, lw=1.4); ax.invert_xaxis()
ax.axhline(proper, color=GREEN, lw=1.0, ls='--'); ax.text(3.4, proper - 0.35, 'proper fit', color=GREEN, fontsize=8, va='top')
ax.axvline(star, color=GREY, lw=0.8, ls=':'); ax.text(star * 1.15, -24.9, '%.3f' % star, fontsize=8, color=GREY, ha='right')
ax.set_xlabel('$\\sigma_2$ (cm)', fontsize=8.5); ax.set_ylabel('ln L', fontsize=8.5); ax.tick_params(labelsize=8)
ax = axs[1]
ax.semilogx(regs, res, color=PUR, lw=1.4, marker='o', ms=3)
ax.axhline(proper, color=GREEN, lw=1.0, ls='--'); ax.axvline(np.exp(thr), color=GREY, lw=0.8, ls=':')
ax.text(np.exp(thr) * 1.15, -16.5, '%.2f' % np.exp(thr), fontsize=8, color=GREY)
ax.set_xlabel('reg_covar (cm$^2$)', fontsize=8.5); ax.set_ylabel('ln L reached', fontsize=8.5); ax.tick_params(labelsize=8)
for a in axs: a.grid(color=GRID, lw=0.4)
fig.subplots_adjust(wspace=0.4)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s40c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s40c1.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.6 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.6 / w))
