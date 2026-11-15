# Problem 51C.1: a test ordered by suspicion. Writes fig/s51c1.png and prints the numbers.
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from scipy.special import expit
from scipy.optimize import brentq

rng = np.random.RandomState(0); n = 4000
s = rng.normal(size=n)                          # recorded symptom score
v = 0.8 * s + 0.6 * rng.normal(size=n)          # laboratory value (variance 1)
hidden = rng.normal(size=n)                     # noise in the doctor's unrecorded impression
susp_rec = s                                    # suspicion from the record only
susp_unr = v + 0.3 * hidden                     # suspicion driven by the unrecorded impression, which tracks v itself
print("true mean of the test value %.3f, sd %.3f, corr(v, s) = %.3f" % (v.mean(), v.std(), np.corrcoef(v, s)[0, 1]))
fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.0), dpi=240, sharey=True)
for ax, susp, name, tit in ((axs[0], susp_rec, 'recorded symptoms', 'symptoms recorded (MAR)'), (axs[1], susp_unr, 'unrecorded impression', 'impression not recorded (MNAR)')):
    z = (susp - susp.mean()) / susp.std()
    a = brentq(lambda a: expit(a + 3.0 * (-z)).mean() - 0.5, -30, 30)      # healthy-looking (low suspicion) -> blank
    blank = rng.uniform(size=n) < expit(a - 3.0 * z)
    seen = ~blank
    b1, b0 = np.polyfit(s[seen], v[seen], 1)
    t1, t0 = np.polyfit(s, v, 1)
    fill_reg = np.where(blank, b0 + b1 * s, v)
    print("%-26s blank %.3f | mean of seen %.3f (true %.3f) | mean-filled mean %.3f sd %.3f | slope on s, seen rows %.3f, true %.3f"
          % (name, blank.mean(), v[seen].mean(), v.mean(), np.where(blank, v[seen].mean(), v).mean(), np.where(blank, v[seen].mean(), v).std(), b1, t1))
    print("%-26s regression fill on s: mean %.3f sd %.3f   mean of the blank values: true %.3f, filled %.3f" % ('', fill_reg.mean(), fill_reg.std(), v[blank].mean(), fill_reg[blank].mean()))
    ax.scatter(s[seen], v[seen], s=3, color=BLUE, alpha=0.5, lw=0, label='ordered')
    ax.scatter(s[blank], v[blank], s=3, color=RED, alpha=0.35, lw=0, label='blank (true value)')
    xs = np.array([-3, 3.]); ax.plot(xs, t0 + t1 * xs, color=GREY, lw=1.1, ls='--', label='true line'); ax.plot(xs, b0 + b1 * xs, color=GREEN, lw=1.4, label='line from ordered')
    ax.set_title(tit, fontsize=8.5, color=NAVY); ax.set_xlabel('recorded symptom score'); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 5.6)
axs[0].set_ylabel('test value'); axs[0].legend(loc='upper left', fontsize=7.6, handlelength=1.2, borderaxespad=0.2, markerscale=2.5)
for ax in axs: ax.tick_params(labelsize=8.3); ax.xaxis.label.set_size(8.5); ax.yaxis.label.set_size(8.5)
fig.tight_layout(pad=0.4)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s51c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s51c1.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))
