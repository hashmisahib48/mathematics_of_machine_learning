# Problem 1C.1: training error, leave-one-out error and test error against the degree. Writes fig/s1c1.png
import sys, os, io, runpy, contextlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from figstyle import *
from numpy.polynomial import Polynomial
import warnings; warnings.simplefilter('ignore', np.exceptions.RankWarning)
buf = io.StringIO()
with contextlib.redirect_stdout(buf): g = runpy.run_path(HERE + '/../../../ch01/code/generalise.py')
xtr, ytr, xte, yte, rm = g['x_tr'], g['y_tr'], g['x_te'], g['y_te'], g['rmse']
n = len(xtr)
degs = list(range(0, 10))
def fit(x, y, k): return Polynomial.fit(x, y, k)
train = [rm(ytr, fit(xtr, ytr, k)(xtr)) for k in degs]
test = [rm(yte, fit(xtr, ytr, k)(xte)) for k in degs]
loo = []
for k in degs:                         # leave-one-out: fit on 11 points, predict the twelfth
    e = [ytr[i] - fit(np.delete(xtr, i), np.delete(ytr, i), k)(xtr[i]) for i in range(n)]
    loo.append(float(np.sqrt(np.mean(np.square(e)))))
print("degree  train RMSE  leave-one-out RMSE  test RMSE")
for k in degs: print(f"{k:6d} {train[k]:11.3f} {loo[k]:19.3f} {test[k]:11.3f}")
print("lowest training RMSE at degree", int(np.argmin(train)), " lowest leave-one-out at degree", int(np.argmin(loo)),
      " lowest test at degree", int(np.argmin(test)))
f11 = fit(xtr, ytr, 11); print("degree 11 (12 knobs, 12 points): training RMSE %.2e" % rm(ytr, f11(xtr)))
print("number of knobs for degree 9:", 10, " training points:", n)
# a 4-fold version needs no more than three extra lines and gives the same winner
fold = np.arange(n) % 4; cv4 = []
for k in degs:
    e = np.concatenate([yte_ - fit(xtr[fold != q], ytr[fold != q], k)(xtr[fold == q]) for q, yte_ in [(q, ytr[fold == q]) for q in range(4)]])
    cv4.append(float(np.sqrt(np.mean(e ** 2))))
print("4-fold RMSE:", np.round(cv4, 3).tolist(), " lowest at degree", int(np.argmin(cv4)))

fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.0), dpi=240, gridspec_kw={'width_ratios': [1.15, 1]})
ax = axs[0]
ax.semilogy(degs, train, '-o', color=BLUE, ms=3.5, lw=1.4, label='training')
ax.semilogy(degs, loo, '--s', color=OR, ms=3.5, lw=1.4, label='leave-one-out')
ax.semilogy(degs, test, ':^', color=GREY, ms=3.5, lw=1.4, label='test (40 new points)')
ax.axvline(int(np.argmin(loo)), color=GREEN, lw=0.9)
ax.set_xlabel('degree', fontsize=8.5); ax.set_ylabel('RMSE', fontsize=8.5); ax.tick_params(labelsize=7.5); ax.legend(fontsize=7.5, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=0.8, handlelength=1.6); ax.grid(color=GRID, lw=0.5)
ax.set_ylim(0.1, 400)
ax = axs[1]
xx = np.linspace(0, 1, 300)
ax.plot(xx, g['truth'](xx), color=LGREY, lw=3, label='true curve')
for k, c, ls in ((int(np.argmin(loo)), GREEN, '-'), (9, RED, '--')):
    ax.plot(xx, fit(xtr, ytr, k)(xx), color=c, lw=1.3, ls=ls, label=f'degree {k}')
ax.scatter(xtr, ytr, s=14, color=BLUE, zorder=5, edgecolor='white', lw=0.4)
ax.set_ylim(-2.5, 2.5); ax.set_xlabel('x', fontsize=8.5); ax.tick_params(labelsize=7.5); ax.legend(fontsize=7.5, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=0.8, handlelength=1.4); ax.grid(color=GRID, lw=0.5)
fig.tight_layout(w_pad=1.0); save(fig, 'fig/s1c1.png')
