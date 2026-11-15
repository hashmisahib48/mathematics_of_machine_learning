# Problem 13C.1: the cliff in epsilon. Writes fig/s13c1.png, verify/c1.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.linear_model import HuberRegressor

x = np.arange(1., 6); y = np.array([3., 4, 6, 7, 20]); X = x[:, None]
def fit(e):
    h = HuberRegressor(epsilon=e, alpha=0, max_iter=5000, tol=1e-12).fit(X, y)
    r = y - h.intercept_ - h.coef_[0] * x; s = h.scale_
    return h.coef_[0], h.intercept_, s, r, int((abs(r) > e * s).sum())
print(" eps  slope  intercept  k  n-k*eps^2   sigma  eps*sigma  |r4|   |r5|  |r5|/sigma  omega5")
for e in (1.20, 1.30, 1.35, 1.38, 1.40, 1.41, 1.42, 1.45, 1.50):
    w, b, s, r, k = fit(e)
    print("%5.2f %6.3f %8.3f %3d %9.3f %8.3f %8.3f %7.3f %6.3f %8.3f %9.3f" % (e, w, b, k, 5 - k * e * e, s, e * s, abs(r[3]), abs(r[4]), abs(r[4]) / s, min(1, e * s / abs(r[4]))))
# the same numbers with the residuals of the eps = 1.35 line held fixed: how far does Eq. 13.3-8 alone move sigma?
w0, b0, s0, r0, k0 = fit(1.35); inn = abs(r0) <= 1.35 * s0
for e in (1.35, 1.40, 1.41):
    w, b, s_, r, k = fit(e); print("eps %.2f: sum of squared residuals inside eps*sigma = %.3f" % (e, (r[abs(r) <= e * s_] ** 2).sum()))
print("\nfixed line of eps = 1.35: sum of inner r^2 = %.4f" % (r0[inn] ** 2).sum())
for e in (1.35, 1.40, 1.45):
    print("  eps %.2f: sigma from Eq. 13.3-8 with k = 2: %.3f   (denominator %.3f)" % (e, np.sqrt((r0[inn] ** 2).sum() / (5 - 2 * e * e)), 5 - 2 * e * e))
print("with k = 2 the denominator n - k eps^2 vanishes at eps = sqrt(5/2) = %.3f" % np.sqrt(2.5))
es = np.linspace(1.15, 1.60, 181); S = []; R5 = []
for e in es:
    w, b, s, r, k = fit(e); S.append(w); R5.append(abs(r[4]) / s)
S = np.array(S); R5 = np.array(R5)
print("steepest part: slope rises from %.3f at eps = 1.40 to %.3f at eps = 1.42; largest step per 0.0025 of eps: %.3f" % (S[np.argmin(abs(es - 1.40))], S[np.argmin(abs(es - 1.42))], np.diff(S).max()))
cross = es[np.argmax(R5 <= es)]; print("|r5|/sigma first falls to eps at eps = %.4f" % cross)

fig, axs = plt.subplots(1, 2, figsize=(4.6, 1.9), dpi=240)
ax = axs[0]; ax.plot(es, S, color=GREEN, lw=1.6); ax.axhline(3.7, color=GREY, lw=0.9, ls='--'); ax.text(1.16, 3.5, 'least squares', fontsize=8.2, color=GREY, va='top')
ax.axvline(1.35, color=PUR, lw=0.9, ls=':'); ax.text(1.355, 0.3 + 1.4, 'default', fontsize=8.2, color=PUR)
ax.set_xlabel('$\\epsilon$', fontsize=9); ax.set_ylabel('Huber slope', fontsize=9); ax.set_ylim(1.4, 4.1)
ax = axs[1]; ax.plot(es, R5, color=BLUE, lw=1.6, label='$|r_5|/\\sigma$'); ax.plot(es, es, color=RED, lw=1.1, ls='--', label='$\\epsilon$')
ax.set_yscale('log'); ax.set_ylim(1, 40); ax.set_yticks([1, 2, 5, 10, 20]); ax.set_yticklabels(['1', '2', '5', '10', '20']); ax.minorticks_off()
ax.axvline(cross, color=GREY, lw=0.8, ls=':'); ax.text(cross + 0.015, 3.6, 'point 5\ninside', fontsize=8.2, color=GREY, va='center')
ax.set_xlabel('$\\epsilon$', fontsize=9); ax.legend(fontsize=8.2, loc='center right', borderpad=0.1, handlelength=1.6, bbox_to_anchor=(1.0, 0.55))
for a in axs: a.grid(color=GRID, lw=0.5); a.tick_params(labelsize=8.2); a.set_axisbelow(True)
fig.tight_layout(w_pad=0.8)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s13c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s13c1.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.6 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.6 / w))
