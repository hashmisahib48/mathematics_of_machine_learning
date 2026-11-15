# Problem 30C.3: the cost of a fit against n (one thread). Writes fig/s30c3.png, prints numbers.
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def best(f, rep=3):
    t = []
    for _ in range(rep): s = time.perf_counter(); f(); t.append(time.perf_counter() - s)
    return min(t)
ns = [500, 1000, 2000, 4000]; tf = []; te = []
for n in ns:
    X = np.random.default_rng(0).normal(size=(n, 3)); y = np.sin(X[:, 0])
    tf.append(best(lambda: GPR(RBF(1.0), alpha=0.01, optimizer=None).fit(X, y)))
    g = GPR(C(1.0) * RBF(1.0), alpha=0.01, optimizer=None).fit(X, y)
    te.append(best(lambda: g.log_marginal_likelihood(g.kernel_.theta, eval_gradient=True), 2) if n <= 2000 else np.nan)
    print(f"n = {n:5d}: fit with a fixed kernel {tf[-1]:7.3f} s    likelihood with gradient {te[-1]:7.3f} s    K needs {8 * n * n / 1e6:7.1f} MB")
tf = np.array(tf); n = np.array(ns, float)
print("log-log slope between n=2000 and 4000:", round(float(np.log(tf[3] / tf[2]) / np.log(2)), 2), "  between 500 and 4000:", round(float(np.log(tf[3] / tf[0]) / np.log(8)), 2))
N = 20000; f = (N / 2000) ** 3
print(f"n = {N}: K needs {8 * N * N / 1e9:.1f} GB (one copy); with the gradient array for 2 hyperparameters {8 * N * N * 3 / 1e9:.1f} GB")
print(f"cubic scaling from n=2000: factor {f:.0f}: fit {tf[2] * f:.0f} s = {tf[2] * f / 60:.1f} min;  one likelihood with gradient {te[2] * f:.0f} s;  50 evaluations {50 * te[2] * f / 3600:.1f} h")
print(f"cubic scaling from n=4000 (factor {(N / 4000) ** 3:.0f}): fit {tf[3] * (N / 4000) ** 3:.0f} s")
print("cost of the subsample route: fit on 2000 (optimiser, ~50 evaluations):", f"{50 * te[2]:.0f} s;  then one factorisation of all {N}: {tf[2] * f / 60:.1f} min")
fig, ax = plt.subplots(figsize=(2.6, 1.7), dpi=240)
ax.loglog(ns, tf, 'o-', color=BLUE, lw=1.4, ms=4, label='measured (1 thread)'); ax.loglog(ns, tf[2] * (n / 2000) ** 3, color=PUR, lw=1.0, ls='--', label='$n^3$')
ax.set_xlabel('training samples $n$'); ax.set_ylabel('fit time (s)'); ax.legend(fontsize=8.5, loc='upper left', handlelength=1.6, borderaxespad=0.1, labelspacing=0.2); ax.set_xticks(ns); ax.set_xticklabels([str(v) for v in ns]); ax.minorticks_off(); ax.tick_params(labelsize=8.5); ax.grid(color=GRID, lw=0.5)
print("min font size:", min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()))
save(fig, 'fig/s30c3.png')
