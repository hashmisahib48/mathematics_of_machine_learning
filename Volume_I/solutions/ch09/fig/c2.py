# Problem 9C.2 figure: mean training R^2 and median test R^2 against the number of features. Writes fig/s9c2.png
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
c = np.load('verify/c2_curve.npy'); p, tr, ft, te = c.T
fig, ax = plt.subplots(figsize=(4.2, 2.6), dpi=240)
ax.plot(p, tr, '-o', color=BLUE, ms=3, lw=1.3, label='training $R^2$ (mean of 400 data sets)')
ax.plot(p, ft, ls='--', color=BLUE, lw=0.8, label='$1-\\sigma^2(n-p-1)/((n-1)\\,\\mathrm{Var}\\,y)$')
ax.plot(p, te, '-s', color=RED, ms=3, lw=1.3, label='test $R^2$ (median)')
ax.axhline(5 / 9, color=GREEN, lw=0.9, ls=':'); ax.text(57, 5 / 9 - 0.17, 'true model, $R^2$ = 0.556', color=GREEN, fontsize=8, ha='right')
ax.axhline(0, color=GREY, lw=0.6); ax.axvline(59, color=GREY, lw=0.8, ls='--'); ax.text(60, -1.5, '$p = n-1$', fontsize=8, color=GREY, rotation=90, va='center')
ax.annotate(('median test $R^2$ = %.0f\nat $p$ = 58 (off scale)' % te[-1]).replace('-', '\u2212'), xy=(57.6, -3.0), xytext=(8, -2.3), fontsize=8, color=RED, ha='left', va='center',
            arrowprops=dict(arrowstyle='->', color=RED, lw=0.8))
ax.set_ylim(-3.1, 1.15); ax.set_xlim(0, 62); ax.set_xlabel('number of features $p$   ($n$ = 60)', fontsize=8.6); ax.set_ylabel('$R^2$', fontsize=8.6)
ax.legend(fontsize=7.6, loc='upper center', bbox_to_anchor=(0.5, -0.24), frameon=False, ncol=1)
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(labelsize=8)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s9c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s9c2.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.2 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.2 / w))
