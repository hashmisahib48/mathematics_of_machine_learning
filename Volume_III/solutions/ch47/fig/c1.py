# Problem 47C.1: one bandwidth for a tight and a wide group. Writes fig/s47c1.png, verify/c1.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from scipy.special import logsumexp
from sklearn.datasets import load_iris

iris = load_iris()
raw = iris.data[:, 2]; sp = iris.target
z = (raw - raw.mean()) / raw.std(); n = len(z)
tight, wide = sp == 0, sp != 0
print("species, n, mean, sd of standardised petal length:")
for k, nm in enumerate(iris.target_names): print(f"  {nm:11s} {np.sum(sp==k):3d} {z[sp==k].mean():7.3f} {z[sp==k].std():6.3f}")
print(f"  setosa      sd {z[tight].std():.3f}   versicolor+virginica sd {z[wide].std():.3f}   ratio {z[wide].std()/z[tight].std():.2f}")
print("rounding step of the data: 0.1 cm =", round(0.1/raw.std(), 3), "scaled units; distinct values of the setosa group:", len(np.unique(z[tight])), "of", tight.sum(), " range", z[tight].min().round(3), z[tight].max().round(3))

D = z[:, None] - z[None, :]                       # pairwise differences x_i - x_j

def logk(d, h): return -d ** 2 / (2 * h ** 2) - 0.5 * np.log(2 * np.pi) - np.log(h)

def loo_fixed(h):                                 # ln f_{-i}(x_i), Gaussian kernel, one h
    lk = logk(D, h); np.fill_diagonal(lk, -np.inf)
    return logsumexp(lk, axis=1) - np.log(n - 1)

def loo_adaptive(h0, pilot=0.2):                  # Abramson: h_j = h0 (f_pilot(x_j)/g)^(-1/2)
    lp = logsumexp(logk(D, pilot), axis=1) - np.log(n)      # pilot ln f at each sample
    hj = h0 * np.exp(-0.5 * (lp - lp.mean()))               # g = geometric mean of the pilot
    lk = logk(D, hj[None, :]); np.fill_diagonal(lk, -np.inf)
    return logsumexp(lk, axis=1) - np.log(n - 1), hj

hs = np.round(np.arange(0.03, 0.80, 0.01), 2)
A = np.array([loo_fixed(h) for h in hs])
mt, mw, ma = A[:, tight].mean(1), A[:, wide].mean(1), A.mean(1)
print("\nleave-one-out mean ln f over each group, best fixed h:")
print(f"  setosa group       best h {hs[mt.argmax()]:.2f}  score {mt.max():.3f}")
print(f"  other two species  best h {hs[mw.argmax()]:.2f}  score {mw.max():.3f}")
print(f"  all 150            best h {hs[ma.argmax()]:.2f}  score {ma.max():.3f}")
for h in (0.11, 0.43):
    i = int(np.argmin(abs(hs - h)))
    print(f"  h = {h}: setosa {mt[i]:.3f}  others {mw[i]:.3f}  all {ma[i]:.3f}")
print("Eq. 47.4-3 with a normal reference: h = 1.0592 sd m^(-1/5)")
for nm, msk in (('setosa', tight), ('others', wide)):
    m = msk.sum(); print(f"  {nm:7s} m = {m:3d} sd {z[msk].std():.3f} -> h = {1.0592*z[msk].std()*m**-0.2:.3f}")

h0s = np.round(np.arange(0.05, 0.60, 0.01), 2)
B = [loo_adaptive(h0)[0] for h0 in h0s]
ba = np.array([b.mean() for b in B]); ib = int(ba.argmax())
bt, bw = B[ib][tight].mean(), B[ib][wide].mean()
hj = loo_adaptive(h0s[ib])[1]
print(f"\nadaptive (Abramson, pilot h = 0.2): best h0 = {h0s[ib]:.2f}  all {ba.max():.3f}  setosa {bt:.3f}  others {bw:.3f}")
print(f"  local bandwidths: setosa {hj[tight].mean():.3f} (range {hj[tight].min():.3f} to {hj[tight].max():.3f}), others {hj[wide].mean():.3f} (range {hj[wide].min():.3f} to {hj[wide].max():.3f})")
print(f"gain of the adaptive estimator over the best fixed h: {ba.max()-ma.max():.3f} nats per iris")

grid = np.linspace(-1.9, 1.9, 600)
def curve(h): return np.exp(logsumexp(logk(grid[:, None] - z[None, :], h), axis=1) - np.log(n))
fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.9), dpi=240)
a = axs[0]; a.plot(hs, mt, color=GREEN, lw=1.3, label='tight group (setosa)'); a.plot(hs, mw, color=PUR, lw=1.3, ls='--', label='wide group (two species)')
a.plot(hs[mt.argmax()], mt.max(), 'o', color=GREEN, ms=4); a.plot(hs[mw.argmax()], mw.max(), 'o', color=PUR, ms=4)
a.set_xlabel('bandwidth $h$ (scaled units)'); a.set_ylabel('LOO mean ln $\\hat f$'); a.set_ylim(-1.6, 0.9); a.legend(fontsize=7.8, loc='upper right', handlelength=1.6)
a = axs[1]; a.plot(grid, curve(0.11), color=GREEN, lw=1.0, label='$h$ = 0.11')
a.plot(grid, curve(0.43), color=PUR, lw=1.0, ls='--', label='$h$ = 0.43')
fa = np.exp(logsumexp(logk(grid[:, None] - z[None, :], hj[None, :]), axis=1) - np.log(n))
a.plot(grid, fa, color=OR, lw=1.3, label='adaptive')
a.plot(z, np.full(n, -0.1), '|', color=BLUE, ms=4, mew=0.4, clip_on=False); a.set_xlabel('petal length (scaled)'); a.set_ylabel('density')
a.set_ylim(0, 1.9); a.legend(fontsize=7.8, loc='upper center', ncol=3, handlelength=1.3, columnspacing=0.7, borderaxespad=0.1)
for a in axs: a.grid(color=GRID, lw=0.5); a.set_axisbelow(True); a.tick_params(labelsize=8.2); a.xaxis.label.set_size(8.4); a.yaxis.label.set_size(8.4)
fig.subplots_adjust(wspace=0.36)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s47c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s47c1.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))
print("highest density: h = 0.11 %.2f, h = 0.43 %.2f, adaptive %.2f" % (curve(0.11).max(), curve(0.43).max(), fa.max()))
for nm, hh in (("h = 0.11", 0.11), ("h = 0.43", 0.43)):
    c = curve(hh); gi = grid < -0.9
    print(f"{nm}: the setosa bump (z < -0.9) has area {c[gi].sum()*(grid[1]-grid[0]):.3f}, peak {c[gi].max():.2f}; dip between the groups (min over -0.9<z<0.0) {c[(grid>-0.9)&(grid<0.0)].min():.3f}")
