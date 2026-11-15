# Problem 43C.2: why Gaussian sources cannot be separated. Writes fig/s43c2.png, verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)      # FastICA often fails to converge on Gaussian data: that is the point

n = 4000; rng = np.random.RandomState(0)
U = (rng.uniform(-1, 1, (n, 2))) * np.sqrt(3)                   # unit-variance uniform sources
G = rng.standard_normal((n, 2))                                 # unit-variance Gaussian sources
M = np.c_[G[:, 0], U[:, 1]]                                     # one Gaussian, one uniform
ang = np.radians(np.arange(0, 181, 2))
def kurt(S): return np.array([(lambda y: np.mean(y**4) / np.mean(y**2)**2 - 3)(S @ np.array([np.cos(a), np.sin(a)])) for a in ang])
kG, kU, kM = kurt(G), kurt(U), kurt(M)
print("kurtosis of the projection on a direction, n = %d (sampling s.d. about %.2f)" % (n, np.sqrt(24 / n)))
for nm, k in (("two Gaussians", kG), ("two uniforms", kU), ("Gaussian + uniform", kM)):
    print(f"{nm:20s} min {k.min():6.2f} at {np.degrees(ang[k.argmin()]):5.1f} deg   max {k.max():6.2f} at {np.degrees(ang[k.argmax()]):5.1f} deg   spread {k.max()-k.min():.2f}")

# covariance after a rotation by 35 degrees is still the identity
th = np.radians(35); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
print("covariance of the rotated Gaussian sources:\n", np.cov((G @ R.T).T).round(3))
# FastICA on mixtures: 40 fresh data sets of 250 samples; how well do the estimates match the true sources?
A = np.array([[2.0, 1.0], [1.0, 1.0]])
def draw(kind, r):
    g = r.standard_normal((250, 2)); u = r.uniform(-1, 1, (250, 2)) * np.sqrt(3)
    return {'gauss': g, 'unif': u, 'mixed': np.c_[g[:, 0], u[:, 1]]}[kind]
for kind, nm in (('gauss', 'two Gaussian sources'), ('unif', 'two uniform sources'), ('mixed', 'one Gaussian, one uniform')):
    w = []
    for rep in range(40):
        S = draw(kind, np.random.RandomState(100 + rep)); est = FastICA(2, random_state=0, whiten="unit-variance").fit_transform(S @ A.T)
        w.append(np.abs(np.corrcoef(est.T, S.T)[:2, 2:]).max(1).min())
    w = np.array(w); print(f"{nm:26s} worst |corr| of the estimates with the true sources over 40 data sets: min {w.min():.3f}  median {np.median(w):.3f}  max {w.max():.3f}")

fig, ax = plt.subplots(figsize=(5.4, 1.9), dpi=240)
deg = np.degrees(ang)
ax.plot(deg, kU, color=BLUE, lw=1.4, label='two uniform sources'); ax.plot(deg, kM, color=GREEN, lw=1.4, label='one Gaussian, one uniform'); ax.plot(deg, kG, color=OR, lw=1.4, label='two Gaussian sources')
ax.axhline(0, color=GREY, lw=0.7); ax.set_xlabel('direction of the projection (degrees)', fontsize=8.4); ax.set_ylabel('kurtosis', fontsize=8.4)
ax.set_xlim(0, 180); ax.set_xticks([0, 45, 90, 135, 180]); ax.tick_params(labelsize=8.2); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(loc='lower left', fontsize=8.2, ncol=1, handlelength=1.4, borderaxespad=0.2, labelspacing=0.25, bbox_to_anchor=(1.01, 0.05))
fig.subplots_adjust(left=0.1, right=0.66, top=0.97, bottom=0.22)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s43c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s43c2.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))
