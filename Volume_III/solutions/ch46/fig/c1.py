# Problem 46C.1: shrinking towards the average variance, with and without standardising. Writes fig/s46c1.png, verify/c1.out
import sys, os, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); sys.path.insert(0, HERE + '/../../../ch46/code'); os.chdir(HERE + '/..')
warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from returns import true_model, draw
from sklearn.covariance import LedoitWolf, empirical_covariance, log_likelihood

Sigma, Theta, sd = true_model(); p = 20; tv = np.diag(Sigma)
def raw_lw(X): m = LedoitWolf().fit(X); return m.covariance_, m.shrinkage_
def std_lw(X):                                   # standardise, shrink the correlation matrix, scale back
    mu, s = X.mean(0), X.std(0); m = LedoitWolf().fit((X - mu) / s); return m.covariance_ * np.outer(s, s), m.shrinkage_
Xte = draw(2000, 99); Ste = empirical_covariance(Xte, assume_centered=True)
X = draw(30, 0); S = empirical_covariance(X); mu = np.trace(S) / p
C1, a1 = raw_lw(X); C2, a2 = std_lw(X)
print("30 days (seed 0): average variance mu = %.3f; sample variances from %.3f to %.3f; true variances from %.3f to %.3f" % (mu, np.diag(S).min(), np.diag(S).max(), tv.min(), tv.max()))
i0, i1 = np.diag(S).argmin(), np.diag(S).argmax()
print("Ledoit-Wolf towards mu I: intensity %.3f" % a1)
for nm, i in (("smallest", i0), ("largest", i1)):
    print("  stock %2d (%s sample variance): true %.3f, sample %.3f, raw Ledoit-Wolf %.3f (check (1-a) S + a mu: %.3f), standardised Ledoit-Wolf %.3f" % (i + 1, nm, tv[i], S[i, i], C1[i, i], (1 - a1) * S[i, i] + a1 * mu, C2[i, i]))
print("standardised: intensity %.3f; its diagonal equals the sample variances: %s" % (a2, np.allclose(np.diag(C2), np.diag(S))))
e = lambda M: ((M - Sigma)**2).sum() / p
R = []
for s in range(100):
    Xs = draw(30, s); Ss = empirical_covariance(Xs); A, _ = raw_lw(Xs); B, _ = std_lw(Xs)
    R.append([e(Ss), e(A), e(B), log_likelihood(Ste, np.linalg.inv(Ss)), log_likelihood(Ste, np.linalg.inv(A)), log_likelihood(Ste, np.linalg.inv(B)),
              *((np.diag(M) - tv) / tv for M in (Ss, A, B))])
R = np.array([np.r_[r[:6], np.concatenate(r[6:])] for r in R]); m = R.mean(0)
print("mean over 100 draws of 30 days:      error      test log-likelihood")
for j, nm in enumerate(("sample covariance", "Ledoit-Wolf, raw", "Ledoit-Wolf, standardised")): print("  %-26s %6.2f %12.2f" % (nm, m[j], m[3 + j]))
rel = m[6:].reshape(3, p)
print("mean relative error of the variance of the 5 stocks with the smallest true variance: sample %.3f, raw %.3f, standardised %.3f" % tuple(rel[:, np.argsort(tv)[:5]].mean(1)))
print("mean relative error of the variance of the 5 stocks with the largest true variance:  sample %.3f, raw %.3f, standardised %.3f" % tuple(rel[:, np.argsort(tv)[-5:]].mean(1)))
print("mean absolute relative error of the 20 variances: sample %.3f, raw %.3f, standardised %.3f" % tuple(np.abs(rel).mean(1)))

o = np.argsort(tv); fig, axs = plt.subplots(1, 2, figsize=(5.4, 1.9), dpi=240)
ax = axs[0]; xs = np.arange(p)
ax.plot(xs, tv[o], '-', color=NAVY, lw=1.3, label='true'); ax.plot(xs, np.diag(S)[o], 'o', color=BLUE, ms=3, label='sample'); ax.plot(xs, np.diag(C1)[o], 's', color=OR, ms=3, label='raw Ledoit-Wolf')
ax.plot(xs, np.diag(C2)[o], '^', color=GREEN, ms=3, label='standardised'); ax.axhline(mu, color=RED, ls='--', lw=1.0); ax.text(0.2, mu + 0.2, 'target $\\hat\\mu$', color=RED, fontsize=8.8)
ax.set_xlabel('stock, in order of true variance'); ax.set_ylabel('variance'); ax.set_xticks([]); ax.legend(loc='upper left', handlelength=1.0, labelspacing=0.2, fontsize=8.8, borderaxespad=0.1, bbox_to_anchor=(0.0, 0.93)); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax = axs[1]
for r_, nm, col, mk in ((rel[0], 'sample', BLUE, 'o'), (rel[1], 'raw Ledoit-Wolf', OR, 's'), (rel[2], 'standardised', GREEN, '^')): ax.plot(xs, r_[o], mk + '-', color=col, ms=3, lw=0.9)
ax.axhline(0, color=NAVY, lw=0.8); ax.set_xlabel('stock, in order of true variance'); ax.set_ylabel('relative error of the variance'); ax.set_xticks([]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.set_title('mean of 100 draws', fontsize=8.8, pad=3)
fig.tight_layout(pad=0.4, w_pad=1.0)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s46c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s46c1.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))
