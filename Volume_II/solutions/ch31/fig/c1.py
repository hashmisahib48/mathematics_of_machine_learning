# Problem 31C.1: equal weights stay equal. Writes fig/s31c1.png, prints numbers.
import sys, os, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from sklearn.datasets import make_moons
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
warnings.simplefilter('ignore', ConvergenceWarning)

Xm, ym = make_moons(100, noise=0.3, random_state=0); Xs = StandardScaler().fit_transform(Xm)
def run(sym):
    n = MLPClassifier((2,), solver='sgd', momentum=0, learning_rate_init=0.5, batch_size=len(Xs), max_iter=1, random_state=0, alpha=0).fit(Xs, ym)
    if sym:
        n.coefs_[0][:] = np.array([[0.4, 0.4], [-0.3, -0.3]]); n.intercepts_[0][:] = 0.1; n.coefs_[1][:] = 0.5; n.intercepts_[1][:] = 0.0
    d, l = [np.linalg.norm(n.coefs_[0][:, 0] - n.coefs_[0][:, 1])], []
    for _ in range(60):
        n.partial_fit(Xs, ym, classes=[0, 1]); d.append(np.linalg.norm(n.coefs_[0][:, 0] - n.coefs_[0][:, 1])); l.append(n.loss_)
    return n, np.array(d), np.array(l)
ns, ds, ls = run(True); nr, dr, lr = run(False)
print("distance between the incoming weight vectors of the two units, after 0, 1, 2, 10, 60 updates")
print("  equal start :", [f"{ds[k]:.1e}" for k in (0, 1, 2, 10, 60)])
print("  random start:", [f"{dr[k]:.3f}" for k in (0, 1, 2, 10, 60)])
print("loss after 60 updates: equal start %.4f, random start %.4f" % (ls[-1], lr[-1]))
print("weights of the equal-start network after 60 updates: W1 =", ns.coefs_[0].round(4).tolist(), " W2 =", ns.coefs_[1][:, 0].round(4).tolist())
# one step by hand: the two error signals are equal because the weights are
W1 = np.array([[0.4, 0.4], [-0.3, -0.3]]); b1 = np.full(2, 0.1); W2 = np.full(2, 0.5); b2 = 0.0
z1 = Xs @ W1 + b1; h = np.maximum(z1, 0); p = 1 / (1 + np.exp(-(h @ W2 + b2))); d2 = p - ym; d1 = d2[:, None] * W2 * (z1 > 0)
G = Xs.T @ d1 / len(Xs)
print("first-step gradient of the two columns of W1:", G[:, 0].round(5).tolist(), G[:, 1].round(5).tolist(), " identical:", bool(np.array_equal(G[:, 0], G[:, 1])))
fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.9), dpi=240)
ax = axs[0]; ax.semilogy(range(61), np.maximum(ds, 1e-17), color=RED, lw=1.6, label='equal start'); ax.semilogy(range(61), dr, color=GREEN, lw=1.6, label='random start')
ax.set_xlabel('update'); ax.set_ylabel('distance of the two units'); ax.set_ylim(1e-18, 20); ax.set_yticks([1e-16, 1e-8, 1]); ax.legend(fontsize=8.5, loc='center right', handlelength=1.6, labelspacing=0.2, borderaxespad=0.1); ax.tick_params(labelsize=8.5); ax.grid(color=GRID, lw=0.5)
ax = axs[1]; ax.plot(range(1, 61), ls, color=RED, lw=1.6, label='equal start'); ax.plot(range(1, 61), lr, color=GREEN, lw=1.6, label='random start')
ax.set_xlabel('update'); ax.set_ylabel('loss'); ax.legend(fontsize=8.5, loc='upper right', handlelength=1.6, labelspacing=0.2, borderaxespad=0.1); ax.tick_params(labelsize=8.5); ax.grid(color=GRID, lw=0.5)
fig.tight_layout(w_pad=0.8)
print("min font size:", min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()))
save(fig, 'fig/s31c1.png')
