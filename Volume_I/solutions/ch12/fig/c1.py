# Problem 12C.1: the floor of the predictive band. Writes fig/s12c1.png
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.linear_model import BayesianRidge

x = np.array([-3., -1, 1, 3]); y = np.array([-2., -2, 1, 3]); X = x[:, None]
r = y - 0.9 * x; sse = float(r @ r)
print("least-squares slope 18/20 = 0.9, residuals", r.round(2).tolist(), " SSE = %.4f" % sse)
print("limit of alpha_ as the copies grow: n/SSE = %.4f ; floor 1/sqrt(alpha) = %.4f" % (4 / sse, (sse / 4) ** 0.5))
ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]; sds = {0.0: [], 3.0: [], 6.0: []}; fl = []; wp6 = []
print("copies   alpha_   floor   sd(0)   sd(3)   sd(6)   weight share of variance at 6")
for k in ks:
    b = BayesianRidge().fit(np.tile(X, (k, 1)), np.tile(y, k)); fl.append(b.alpha_ ** -0.5)
    for xs in sds: sds[xs].append(float(b.predict([[xs]], return_std=True)[1][0]))
    v6 = sds[6.0][-1] ** 2; wp6.append(1 - (1 / b.alpha_) / v6)
    print(f"{k:6d} {b.alpha_:8.4f} {fl[-1]:7.4f} {sds[0.0][-1]:7.4f} {sds[3.0][-1]:7.4f} {sds[6.0][-1]:7.4f} {wp6[-1]:10.3f}")
b64 = BayesianRidge().fit(np.tile(X, (64, 1)), np.tile(y, 64)); m0, s0 = b64.predict([[0.0]], return_std=True)
print("64 copies: sd at the centre %.4f = 1/sqrt(alpha_) %.4f (equal: the weight part is x*^2 S = 0 there)" % (s0[0], b64.alpha_ ** -0.5))
print("ratio of the band at 6 to the floor, 1 copy and 1024 copies: %.3f, %.3f" % (sds[6.0][0] / fl[0], sds[6.0][-1] / fl[-1]))
print("effect of extra information: the mean of m readings has floor 1/sqrt(m alpha_):", [round(float(fl[-1] / m ** 0.5), 3) for m in (1, 2, 4)])
print("slope sd for 1 and 1024 copies: %.4f, %.4f" % (BayesianRidge().fit(X, y).sigma_[0, 0] ** 0.5, BayesianRidge().fit(np.tile(X, (1024, 1)), np.tile(y, 1024)).sigma_[0, 0] ** 0.5))

fig, ax = plt.subplots(figsize=(3.4, 2.05), dpi=240)
for (xs, v), c, ls in zip(sds.items(), (BLUE, OR, GREEN), ('-', '--', '-.')): ax.plot(ks, v, color=c, lw=1.5, ls=ls, label=f'$x_*$ = {xs:.0f}')
ax.axhline((sse / 4) ** 0.5, color=RED, lw=1.1, ls=':'); ax.text(1.2, (sse / 4) ** 0.5 - 0.03, 'floor 0.671', color=RED, fontsize=8.5, va='top')
ax.set_xscale('log'); ax.set_xlabel('number of copies of the four points', fontsize=8.5); ax.set_ylabel('predictive s.d.', fontsize=8.5); ax.tick_params(labelsize=8.5)
ax.set_ylim(0.4, 1.4); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); ax.legend(fontsize=8.5, loc='upper right', borderpad=0.2, handlelength=1.8, labelspacing=0.2)
fig.tight_layout(pad=0.4)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s12c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s12c1.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 3.5 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 3.5 / w))
