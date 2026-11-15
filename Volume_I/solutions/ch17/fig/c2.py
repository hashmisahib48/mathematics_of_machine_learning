# Problem 17C.2: a perceptron that does not settle. Writes fig/s17c2.png, verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.datasets import make_blobs
from sklearn.linear_model import Perceptron, SGDClassifier, LogisticRegression

X, y = make_blobs(n_samples=100, centers=[(-1.0, -0.7), (1.0, 0.7)], cluster_std=1.3, random_state=3)
def curve(m, epochs=40):
    acc, mis = [], []
    for k in range(epochs):
        m.partial_fit(X, y, classes=[0, 1]); acc.append(m.score(X, y))
    return np.array(acc)
runs = {
 'Perceptron (the book)': Perceptron(random_state=0, tol=None, max_iter=1),
 'perceptron loss, average=True': SGDClassifier(loss='perceptron', learning_rate='constant', eta0=1.0, penalty=None, average=True, random_state=0),
 'log loss, alpha=0.01, optimal rate': SGDClassifier(loss='log_loss', alpha=0.01, random_state=0),
 'perceptron loss, rate 1/sqrt(t)': SGDClassifier(loss='perceptron', learning_rate='invscaling', eta0=1.0, power_t=0.5, penalty=None, random_state=0),
}
R = {k: curve(m) for k, m in runs.items()}
print("epochs 1..40; training accuracy after each epoch")
for k, a in R.items():
    print(f"{k:38s} min {a.min():.2f} max {a.max():.2f} last {a[-1]:.2f}   epochs 21-40: sd {a[20:].std():.3f}  range {a[20:].max() - a[20:].min():.2f}")
print("number of epochs (of 40) with accuracy 1.00 for the Perceptron:", int((R['Perceptron (the book)'] == 1).sum()))
lr = LogisticRegression(C=1e6, max_iter=100000).fit(X, y); print("best linear fit, logistic regression C=1e6: training accuracy", lr.score(X, y), " mistakes", int((lr.predict(X) != y).sum()))
# is any line perfect?  scan 3600 directions, best threshold for each
best = 0
for th in np.linspace(0, np.pi, 1800, endpoint=False):
    p = X @ np.array([np.cos(th), np.sin(th)]); o = np.argsort(p); ys = y[o]
    c1 = np.cumsum(ys == 0); c2 = np.cumsum(ys == 1); n0, n1 = c1[-1], c2[-1]
    ok = np.r_[n1, c1 + (n1 - c2)]; best = max(best, ok.max(), (100 - ok).max())
print("best of 1800 directions x all thresholds:", int(best), "of 100 correct")
w = Perceptron(random_state=0, tol=None, max_iter=1); mis = []
for k in range(40):
    w.partial_fit(X, y, classes=[0, 1]); mis.append(int((w.predict(X) != y).sum()))
print("mistakes after each epoch, Perceptron:", mis)
print("norm of w after epochs 10, 20, 30, 40: ", end="")
w = Perceptron(random_state=0, tol=None, max_iter=1); nn = []
for k in range(40): w.partial_fit(X, y, classes=[0, 1]); nn.append(np.linalg.norm(w.coef_))
print(np.round([nn[9], nn[19], nn[29], nn[39]], 2))

fig, ax = plt.subplots(figsize=(5.4, 1.95), dpi=240)
cols = [RED, BLUE, GREEN, PUR]
for (k, a), c_ in zip(R.items(), cols): ax.plot(range(1, 41), a, color=c_, lw=1.3, label=k)
ax.set_xlabel('epoch', fontsize=9.5); ax.set_ylabel('training accuracy', fontsize=9.5); ax.tick_params(labelsize=9); ax.grid(color=GRID, lw=0.5)
ax.set_ylim(0.5, 0.95); ax.legend(fontsize=8.2, ncol=2, loc='lower center', borderaxespad=0.2, handlelength=1.6, columnspacing=1.0)
fig.tight_layout()
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s17c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s17c2.png') as im: w_ = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w_, fs, fs * 4.9 / w_))
