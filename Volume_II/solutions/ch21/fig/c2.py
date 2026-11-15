# Problem 21C.2: filter factors of the Gram matrix. Writes fig/s21c2.png, verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.metrics.pairwise import rbf_kernel

rng = np.random.default_rng(4); xs = np.sort(rng.uniform(0, 6, 28)); alpha = 0.1
print("28 sorted points of the noisy sine, alpha = %.1f; trace of K = %d (every diagonal entry is 1)" % (alpha, len(xs)))
fig, axs = plt.subplots(1, 2, figsize=(4.9, 1.9), dpi=240, gridspec_kw={'width_ratios': [1, 1]})
for g, c_ in ((0.05, PUR), (0.5, GREEN), (5, OR)):
    lam = np.sort(np.linalg.eigvalsh(rbf_kernel(xs[:, None], gamma=g)))[::-1]; lam = np.maximum(lam, 0); f = lam / (lam + alpha)
    print("gamma = %-4g: lambda_1 = %.2f, lambda_2 = %.3f, lambda_5 = %.2e, lambda_10 = %.2e, lambda_20 = %.2e"
          % (g, lam[0], lam[1], lam[4], lam[9], lam[19]))
    print("            eigenvalues above alpha: %d; filter factor above 0.5: %d; above 0.9: %d; df = sum of filter factors = %.2f; first six filter factors %s"
          % ((lam > alpha).sum(), (f > 0.5).sum(), (f > 0.9).sum(), f.sum(), np.round(f[:6], 3).tolist()))
    print("            share of the trace carried by the top 3 eigenvalues: %.3f" % (lam[:3].sum() / lam.sum()))
    axs[0].semilogy(range(1, 29), np.maximum(lam, 1e-16), color=c_, lw=1.4, label='$\\gamma$ = %g' % g)
    axs[1].plot(range(1, 29), f, color=c_, lw=1.4, marker='o', ms=2.5)
axs[0].axhline(alpha, color=RED, lw=0.9, ls='--'); axs[0].text(28, 0.2, '$\\alpha$ = 0.1', color=RED, fontsize=8.5, ha='right')
axs[0].set_ylim(1e-12, 1e2); axs[0].set_ylabel('eigenvalue $\\lambda_j$', fontsize=9); axs[0].legend(fontsize=8.5, loc='lower left')
axs[1].set_ylabel('filter factor $\\lambda_j/(\\lambda_j+\\alpha)$', fontsize=8.5); axs[1].axhline(0.5, color=GREY, lw=0.7, ls=':')
for ax in axs: ax.set_xlabel('index $j$', fontsize=9); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(labelsize=8.5)
fig.tight_layout(w_pad=0.8)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s21c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s21c2.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.7 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.7 / w))
