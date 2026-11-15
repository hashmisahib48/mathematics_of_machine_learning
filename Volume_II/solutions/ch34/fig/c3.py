# Problem 34C.3: six labels from the top of the cloud. Writes fig/s34c3.png and prints the numbers.
import sys, os, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
warnings.simplefilter('ignore')
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
C = [[-1.5, 0], [1.5, 0]]
def models(X, y, lab):
    yy = -np.ones(len(y), int); yy[lab] = y[lab]
    lr = LogisticRegression().fit(X[lab], y[lab])
    ls = LabelSpreading(kernel='knn', n_neighbors=10, alpha=0.9, max_iter=5000, tol=1e-6).fit(X, yy)
    st = SelfTrainingClassifier(LogisticRegression(), threshold=0.9, max_iter=None).fit(X, yy)
    return lr, ls, st
def pick(X, y, how, rng):
    out = []
    for k in (0, 1):
        ix = np.where(y == k)[0]
        out += list(ix[np.argsort(-X[ix, 1])[:3]]) if how == 'top' else list(rng.choice(ix, 3, replace=False))
    return np.array(out)
Xt, yt = make_blobs(6000, centers=C, cluster_std=1.6, random_state=99)
print("best possible on 6000 new samples: %.3f" % np.mean((Xt[:, 0] > 0) == yt))
rng = np.random.default_rng(0)
for how in ('top', 'random'):
    A = []
    for s in range(100):
        X, y = make_blobs(300, centers=C, cluster_std=1.6, random_state=2000 + s)
        lr, ls, st = models(X, y, pick(X, y, how, rng))
        A.append([lr.score(Xt, yt), ls.score(Xt, yt), st.score(Xt, yt), ls.score(X, y)])
    A = np.array(A); print("labels from the %-6s mean over 100 draws: logistic %.3f  spreading %.3f  self-training %.3f   (spreading, transductive %.3f)  spreading better than logistic in %d draws"
                            % (how, *A.mean(0), (A[:, 1] > A[:, 0]).sum()))
X, y = make_blobs(300, centers=C, cluster_std=1.6, random_state=4)      # the data of the problem figure
lab = pick(X, y, 'top', rng); lr, ls, st = models(X, y, lab)
print("figure draw: logistic %.3f  spreading %.3f (transductive on the 300: %.3f); labelled x2 values %s; sd of x2 over the cloud %.2f"
      % (lr.score(Xt, yt), ls.score(Xt, yt), ls.score(X, y), np.round(X[lab, 1], 2), X[:, 1].std()))
print("logistic coefficients on the six labels:", lr.coef_.round(2), " intercept %.2f" % lr.intercept_[0])
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.0), dpi=240)
g = np.linspace(-5, 5, 200); G = np.c_[np.repeat(g, 200), np.tile(g, 200)]
for ax, m, ttl in ((axs[0], lr, 'logistic regression, six labels'), (axs[1], ls, 'label spreading, all 300 samples')):
    ax.contourf(g, g, m.predict(G).reshape(200, 200).T, levels=[-.5, .5, 1.5], colors=['#DCE8F8', '#FBE9CF'], zorder=0)
    for k, mk, cc in ((0, 'o', BLUE), (1, 's', OR)):
        ax.scatter(X[y == k, 0], X[y == k, 1], s=4, marker=mk, color=cc, lw=0, alpha=0.55, zorder=2)
        l = lab[y[lab] == k]; ax.scatter(X[l, 0], X[l, 1], s=36, marker=mk, color=cc, edgecolor=NAVY, lw=1.0, zorder=6)
    ax.axvline(0, color=GREY, ls=':', lw=1.0); ax.set_xlim(-5.8, 5.8); ax.set_ylim(-5.8, 5.8); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(ttl, fontsize=8.8, pad=3)
    for s_ in ax.spines.values(): s_.set_visible(False)
fig.subplots_adjust(wspace=0.05, left=0.01, right=0.99, top=0.9, bottom=0.01)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s34c3.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
w = Image.open('fig/s34c3.png').size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))
