# Problem 25C.2: does the forest overfit with more trees? Writes fig/s25c2.png, verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split

d = load_breast_cancer(); Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=0.25, random_state=0, stratify=d.target)
n = len(yte); NB = 300
def curve(seed):
    f = RFC(NB, random_state=seed).fit(Xtr, ytr); Pc = np.cumsum([t.predict_proba(Xte) for t in f.estimators_], axis=0)
    return np.array([(Pc[b].argmax(1) != yte).mean() for b in range(NB)])
E = np.array([curve(s) for s in range(25)]); e0 = E[0]
print("the book's forest (seed 0): error at B = 1: %.4f (%d patients), minimum %.4f (%d patients) at B = %d, at B = 300: %.4f (%d patients)" % (e0[0], round(e0[0] * n), e0.min(), round(e0.min() * n), e0.argmin() + 1, e0[-1], round(e0[-1] * n)))
print("one patient = %.4f; standard error of an error rate of 0.05 on %d patients: %.4f" % (1 / n, n, np.sqrt(0.05 * 0.95 / n)))
Bs = [1, 5, 10, 20, 40, 100, 300]
print("mean test error of 25 forests, with the smallest and largest of the 25:")
for B in Bs: print("  B = %3d: mean %.4f  min %.4f  max %.4f" % (B, E[:, B - 1].mean(), E[:, B - 1].min(), E[:, B - 1].max()))
am = E.argmin(1) + 1; print("position of the minimum of each of the 25 curves: %s" % np.sort(am).tolist())
print("average over the 25 curves of their own minimum: %.4f   average error at B = 300: %.4f" % (E.min(1).mean(), E[:, -1].mean()))
print("share of the 25 curves whose error at B = 300 is higher than at B = 5: %.2f;  mean error over B = 100..300: %.4f vs mean over B = 40..100: %.4f" % ((E[:, -1] > E[:, 4]).mean(), E[:, 99:].mean(), E[:, 39:100].mean()))
print("mean over curves: error at B = 5 is %.4f, at B = 300 %.4f; Eq. 25.3-8 says the mean can only fall" % (E[:, 4].mean(), E[:, -1].mean()))

fig, axs = plt.subplots(1, 2, figsize=(5.4, 1.95), dpi=240); Bg = np.arange(1, NB + 1)
ax = axs[0]; ax.plot(Bg, e0 * n, color=BLUE, lw=1.1); ax.set_xscale('log'); ax.set_title('the forest of the problem', fontsize=8.6, pad=3); ax.set_ylabel('test errors (of 143)', fontsize=8.6)
ax.plot([e0.argmin() + 1], [e0.min() * n], 'o', color=OR, ms=5); ax.set_xticks([1, 10, 100]); ax.set_xticklabels(['1', '10', '100'])
ax = axs[1]
for e in E[:12]: ax.plot(Bg, e * n, color=GREY, lw=0.5, alpha=0.5)
ax.plot(Bg, E.mean(0) * n, color=GREEN, lw=1.8, label='mean of 25 forests'); ax.set_xscale('log'); ax.set_title('25 forests with other seeds', fontsize=8.6, pad=3); ax.set_xticks([1, 10, 100]); ax.set_xticklabels(['1', '10', '100']); ax.legend(loc='upper right', fontsize=8.6, borderaxespad=0.1)
for a in axs: a.set_xlabel('trees $B$', fontsize=8.6); a.tick_params(labelsize=8.6); a.set_ylim(0, 22); a.grid(color=GRID, lw=0.5); a.set_axisbelow(True)
fig.subplots_adjust(wspace=0.28)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s25c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s25c2.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))
