# Problem 45C.2: spiky and flat vectors under a Gaussian and a very sparse matrix. Writes fig/s45c2.png, verify/c2.out
import sys, os, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.random_projection import SparseRandomProjection

p, k, s, reps = 400, 10, 20, 20000
flat = np.ones(p) / np.sqrt(p); spiky = np.zeros(p); spiky[0] = 1.0; two = np.zeros(p); two[[0, 1]] = [1, -1]
rng = np.random.RandomState(0)
def draw(kind, x):                       # the squared length of Rx for many matrices; entries have variance 1/k
    x = x[x != 0]                        # only the columns where x is not zero matter
    if kind == 'gauss': R = rng.normal(0, 1 / np.sqrt(k), (reps, k, len(x)))
    else:
        R = rng.choice([-1., 0., 1.], p=[1 / (2 * s), 1 - 1 / s, 1 / (2 * s)], size=(reps, k, len(x))) * np.sqrt(s / k)
    return ((R @ x)**2).sum(1) / (x @ x)
form = lambda x, s_: np.sqrt((2 + (s_ - 3) * (x**4).sum() / (x @ x)**2) / k)
Q = {}
print("Eq. 45.3-5 for p = %d, k = %d:  sd of q = sqrt((2 + (s-3) Sum x^4/|x|^4)/k)" % (p, k))
print("vector          Sum x^4/|x|^4   Gaussian (s=3)  formula / simulation     sparse s=%d: formula / simulation   share of q = 0" % s)
for nm, x in (('flat', flat), ('spiky (1 entry)', spiky), ('2 spikes', two)):
    r4 = (x**4).sum() / (x @ x)**2
    qg, qs = draw('gauss', x), draw('sparse', x); Q[nm] = (qg, qs)
    print("%-15s %10.4f   %15.4f / %.4f   %18.4f / %.4f   %10.4f" % (nm, r4, np.sqrt(2 / k), qg.std(), form(x, s), qs.std(), (qs == 0).mean()))
print("probability that the column of the single spike has all %d entries zero: (1 - 1/s)^k = %.4f" % (k, (1 - 1 / s)**k))
print("mean of q in every case:", {nm: (round(float(v[0].mean()), 3), round(float(v[1].mean()), 3)) for nm, v in Q.items()})
print("k needed for sd(q) <= 0.2 (Eq. 45.3-5): Gaussian %d;  sparse, flat %d;  sparse, spiky %d;  sparse, 2 spikes %d" % tuple(
    int(np.ceil((2 + (sv - 3) * r) / 0.04)) for sv, r in ((3, 0), (s, 1 / p), (s, 1), (s, 0.5))))
print("cost of one projected sample is proportional to k p / s: Gaussian k = 50: %.1f p;  sparse k = 475: %.2f p;  sparse k = 52 (flat vectors): %.2f p" % (50, 475 / s, 52 / s))
print("s = 3 (Achlioptas): spiky sd %.4f, same as Gaussian %.4f;   s = 20 costs 1/%d of the multiplications" % (form(spiky, 3), np.sqrt(2 / k), s))
with warnings.catch_warnings():
    warnings.simplefilter('ignore'); sk = np.array([(SparseRandomProjection(k, density=1 / s, random_state=r).fit_transform(spiky[None])**2).sum() for r in range(3000)])
print("check with SparseRandomProjection(density = 1/20), spiky vector, 3000 matrices: sd %.3f, share of zeros %.3f" % (sk.std(), (sk == 0).mean()))

fig, axs = plt.subplots(1, 4, figsize=(5.4, 1.5), dpi=240, sharey=True); bins = np.linspace(0, 4, 41)
for ax, (nm, kind, col) in zip(axs, (('flat', 0, BLUE), ('spiky (1 entry)', 0, BLUE), ('flat', 1, OR), ('spiky (1 entry)', 1, OR))):
    v = Q[nm][kind]; ax.hist(np.clip(v, 0, 3.98), bins=bins, color=col, weights=np.ones(len(v)) / len(v), edgecolor='white', lw=0.2)
    ax.axvline(1, color=NAVY, lw=0.8); ax.set_xticks([0, 1, 2, 3, 4]); ax.set_yticks([0, 0.2, 0.4, 0.6]); ax.set_ylim(0, 0.65)
    ax.set_title(('Gaussian' if kind == 0 else 'sparse $s=20$') + ',\n' + nm.split(' ')[0] + ' vector', fontsize=8.8, pad=2); ax.set_xlabel('$q$', labelpad=1)
axs[0].set_ylabel('share of matrices')
fig.tight_layout(pad=0.3, w_pad=0.5)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s45c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s45c2.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))
