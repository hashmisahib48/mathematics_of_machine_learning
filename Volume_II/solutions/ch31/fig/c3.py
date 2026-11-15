# Problem 31C.3: which epoch? Writes fig/s31c3.png, prints numbers.
import sys, os, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from sklearn.datasets import make_moons
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
warnings.simplefilter('ignore', ConvergenceWarning)

Xd, yd = make_moons(400, noise=0.5, random_state=3); Xd = StandardScaler().fit_transform(Xd)
Xtr, ytr, Xva, yva = Xd[:60], yd[:60], Xd[60:], yd[60:]
net = MLPClassifier((100,), solver='adam', alpha=0, random_state=0, learning_rate_init=0.01, batch_size=60)
tl, vl, va = [], [], []
for ep in range(300):
    net.partial_fit(Xtr, ytr, classes=[0, 1]); tl.append(log_loss(ytr, net.predict_proba(Xtr))); vl.append(log_loss(yva, net.predict_proba(Xva))); va.append(net.score(Xva, yva))
tl, vl, va = map(np.array, (tl, vl, va)); bi = int(vl.argmin()) + 1
print(f"{len(ytr)} training and {len(yva)} validation samples; standard error of an accuracy near 0.8 on 340 samples: {np.sqrt(0.8 * 0.2 / 340):.3f}")
print(f"minimum validation log loss {vl.min():.3f} at epoch {bi}; training loss there {tl[bi - 1]:.3f}; validation accuracy there {va[bi - 1]:.3f}")
print(f"epochs with validation loss within 0.05 of the minimum: {int((vl <= vl.min() + 0.05).sum())} (first {int(np.argmax(vl <= vl.min() + 0.05)) + 1}, last {len(vl) - int(np.argmax((vl <= vl.min() + 0.05)[::-1]))})")
print(f"epoch 300: training loss {tl[-1]:.3f}, validation loss {vl[-1]:.3f}, validation accuracy {va[-1]:.3f}")
print(f"best validation accuracy {va.max():.3f}; epochs within one standard error (0.022) of it: {int((va >= va.max() - 0.022).sum())} (first {int(np.argmax(va >= va.max() - 0.022)) + 1}, last {len(va) - int(np.argmax((va >= va.max() - 0.022)[::-1]))})")
print("the training loss keeps falling: monotone after epoch 20:", bool(np.all(np.diff(tl[20:]) < 1e-3)), " (largest rise %.4f)" % np.diff(tl[20:]).max())
stops, tests = [], []
for s in range(20):
    m = MLPClassifier((100,), solver='adam', alpha=0, random_state=s, learning_rate_init=0.01, batch_size=54, early_stopping=True, n_iter_no_change=10, max_iter=300).fit(Xtr, ytr)
    stops.append(m.n_iter_ - 11); tests.append(m.score(Xva, yva))
print("default early_stopping (validation_fraction 0.1 = 6 samples, n_iter_no_change 10), 20 seeds:")
print("  epoch of the best validation score: min %d, median %d, max %d" % (min(stops), np.median(stops), max(stops)))
print("  accuracy on the 340 held-out samples: min %.3f, mean %.3f, max %.3f" % (min(tests), np.mean(tests), max(tests)))
fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.9), dpi=240)
ax = axs[0]; ax.plot(range(1, 301), tl, color=BLUE, lw=1.4, label='training'); ax.plot(range(1, 301), vl, color=GREY, lw=1.4, ls='--', label='validation'); ax.axvline(bi, color=PUR, lw=1)
ax.text(bi + 8, 1.95, f'min at {bi}', fontsize=8.5, color=PUR, va='top'); ax.set_ylim(0, 2.1); ax.set_xlabel('epoch'); ax.set_ylabel('log loss'); ax.legend(fontsize=8.5, loc='center right', handlelength=1.6, labelspacing=0.2, borderaxespad=0.1); ax.tick_params(labelsize=8.5); ax.grid(color=GRID, lw=0.5)
ax = axs[1]; ax.hist(stops, bins=np.arange(0.5, 14.5, 1), color=LB, edgecolor=BLUE, lw=0.8); ax.axvline(bi, color=PUR, lw=1); ax.set_xlim(0, 14)
ax.set_xlabel('epoch where default early stopping stops'); ax.set_ylabel('seeds (of 20)'); ax.tick_params(labelsize=8.5)
fig.tight_layout(w_pad=0.8)
print("min font size:", min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()))
save(fig, 'fig/s31c3.png')
