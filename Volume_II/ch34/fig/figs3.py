# Figures of Chapter 34, part 3: when unlabelled data hurt. Run: python3 fig/figs3.py
import sys, os, json, warnings, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../lib'); os.chdir(HERE + '/..')
from figstyle import *
import matplotlib.text as mtext
from sklearn.datasets import make_blobs, load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier
warnings.simplefilter('ignore', ConvergenceWarning); warnings.simplefilter('ignore', RuntimeWarning)
plt.rcParams.update({'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.labelsize': 9.5, 'legend.fontsize': 9})
NUM = {}; CHK = {}
CL = [BLUE, OR]; MK = ['o', 's']
t0 = time.time()

def sv(fig, name, printed_px):
    fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig, name)
    from PIL import Image
    w = Image.open(name).size[0] / 240
    CHK[name] = {'fig_in': round(w, 2), 'printed_in': round(printed_px / 96, 2), 'min_font_design': round(fs, 1),
                 'min_font_printed': round(fs * printed_px / 96 / w, 1)}

# ---------------------------------------------------------------- f3101 : overlapping blobs, 6 labels, 200 draws
C0 = [[-1.5, 0], [1.5, 0]]
def draw(seed):
    X, y = make_blobs(400, centers=C0, cluster_std=1.6, random_state=seed)
    rng = np.random.default_rng(seed)
    idx = np.r_[rng.choice(np.where(y == 0)[0], 3, replace=False), rng.choice(np.where(y == 1)[0], 3, replace=False)]
    yy = -np.ones(400, int); yy[idx] = y[idx]
    return X, y, idx, yy
D_st, D_ls, SUP = [], [], []
Xt, yt = make_blobs(6000, centers=C0, cluster_std=1.6, random_state=12345)
for seed in range(200):
    X, y, idx, yy = draw(seed)
    s = LogisticRegression().fit(X[idx], y[idx]).score(Xt, yt)
    st = SelfTrainingClassifier(LogisticRegression(), threshold=0.9).fit(X, yy).score(Xt, yt)
    ls = LabelSpreading(kernel='rbf', gamma=1, alpha=0.5, max_iter=300).fit(X, yy).score(Xt, yt)
    SUP.append(s); D_st.append(st - s); D_ls.append(ls - s)
D_st, D_ls, SUP = map(np.array, (D_st, D_ls, SUP))
NUM['hurt'] = {'draws': 200, 'sup_mean': float(SUP.mean()), 'st_mean_change': float(D_st.mean()), 'st_worse': float((D_st < 0).mean()), 'st_better': float((D_st > 0).mean()),
               'st_min': float(D_st.min()), 'st_max': float(D_st.max()), 'ls_mean_change': float(D_ls.mean()), 'ls_worse': float((D_ls < 0).mean()), 'ls_better': float((D_ls > 0).mean())}
# the Bayes classifier of two equal spherical Gaussians is the line x = 0
NUM['hurt']['bayes_acc'] = float(((Xt[:, 0] > 0).astype(int) == yt).mean())
print(NUM['hurt'], round(time.time() - t0, 1))
neg = np.where(D_st < 0)[0]; pick = int(neg[np.argmin(abs(D_st[neg] - D_st[neg].mean()))])
X, y, idx, yy = draw(pick)
sup = LogisticRegression().fit(X[idx], y[idx]); st = SelfTrainingClassifier(LogisticRegression(), threshold=0.9).fit(X, yy)
NUM['hurt']['pick'] = {'seed': pick, 'sup_acc': float(sup.score(Xt, yt)), 'st_acc': float(st.score(Xt, yt)), 'n_pseudo': int((st.labeled_iter_ > 0).sum()), 'n_iter': int(st.n_iter_)}
fig, axs = plt.subplots(1, 2, figsize=(5.7, 2.3), dpi=240, gridspec_kw={'width_ratios': [1.1, 1]})
ax = axs[0]
for k in (0, 1):
    m = (yy == -1) & (y == k); ax.scatter(X[m, 0], X[m, 1], s=5, marker=MK[k], color=CL[k], lw=0, alpha=0.5, zorder=2)
for i in idx: ax.scatter(X[i, 0], X[i, 1], s=44, marker=MK[y[i]], color=CL[y[i]], edgecolor=NAVY, lw=1.1, zorder=6)
xs = np.array([-5, 5.])
def line(model): w = model.coef_[0]; b = model.intercept_[0]; return -(w[0] * xs + b) / w[1]
ax.plot([0, 0], [-5, 5], color=GREY, lw=1.0, ls=':', zorder=3); ax.plot(xs, line(sup), color=GREEN, lw=1.4, ls='--', zorder=4); ax.plot(xs, line(st.estimator_), color=PUR, lw=1.4, ls='-', zorder=4)
ax.set_xlim(-5.2, 5.2); ax.set_ylim(-5, 5); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
for s_ in ax.spines.values(): s_.set_visible(False)
ax.plot([], [], color=GREY, ls=':', lw=1.0, label='best possible'); ax.plot([], [], color=GREEN, ls='--', lw=1.4, label='6 labels only'); ax.plot([], [], color=PUR, lw=1.4, label='self-training')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.0), ncol=1, fontsize=8.6, handlelength=1.8, labelspacing=0.15, borderpad=0.1)
ax = axs[1]; bins = np.linspace(-0.14, 0.09, 24)
ax.hist(D_st, bins=bins, histtype='stepfilled', color=PUR, alpha=0.30, lw=0); ax.hist(D_st, bins=bins, histtype='step', color=PUR, lw=1.3, label='self-training')
ax.hist(D_ls, bins=bins, histtype='step', color=GREEN, lw=1.3, ls='--', label='LabelSpreading')
ax.axvline(0, color=GREY, lw=0.9); ax.set_xlabel('test accuracy minus that of the 6-label model'); ax.set_ylabel('draws (of 200)')
ax.legend(loc='upper left', fontsize=8.6, handlelength=1.8, labelspacing=0.15, borderpad=0.1); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(w_pad=1.0)
sv(fig, 'f3101.png', 470)
print('hurt fig', round(time.time() - t0, 1))

json.dump(NUM, open('verify/fig_numbers3.json', 'w'), indent=1); json.dump(CHK, open('verify/fig_textsize3.json', 'w'), indent=1)
print(json.dumps(CHK)); print(round(time.time() - t0, 1), 's')
