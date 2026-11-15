# Figures of Chapter 34, part 2: the gamma gallery and the three knobs on the digits. Run: python3 fig/figs2.py
import sys, os, json, warnings, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../lib'); sys.path.insert(0, HERE + '/../code'); os.chdir(HERE + '/..')
from figstyle import *
import matplotlib.text as mtext
from matplotlib.colors import LinearSegmentedColormap
from sklearn.datasets import make_moons, load_digits
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.exceptions import ConvergenceWarning
from scratch import GraphSSL
warnings.simplefilter('ignore', ConvergenceWarning); warnings.simplefilter('ignore', RuntimeWarning)
plt.rcParams.update({'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.labelsize': 9.5, 'legend.fontsize': 9})
NUM = {}; CHK = {}
CL = [BLUE, OR]; MK = ['o', 's']
CM = LinearSegmentedColormap.from_list('bo', ['#C9DCF5', '#FFFFFF', '#FBE3BC'])
t0 = time.time()

def sv(fig, name, printed_px):
    fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig, name)
    from PIL import Image
    w = Image.open(name).size[0] / 240
    CHK[name] = {'fig_in': round(w, 2), 'printed_in': round(printed_px / 96, 2), 'min_font_design': round(fs, 1),
                 'min_font_printed': round(fs * printed_px / 96 / w, 1)}

# ---------------------------------------------------------------- f392 : gamma gallery on the two moons
X, y = make_moons(200, noise=0.1, random_state=1)
rng = np.random.default_rng(3)
lab = np.r_[rng.choice(np.where(y == 0)[0], 2, replace=False), rng.choice(np.where(y == 1)[0], 2, replace=False)]
yy = -np.ones(200, int); yy[lab] = y[lab]; unl = yy == -1
gx, gy = np.meshgrid(np.linspace(-1.5, 2.5, 160), np.linspace(-1.0, 1.5, 120))
G = np.c_[gx.ravel(), gy.ravel()]
fig, axs = plt.subplots(1, 4, figsize=(5.7, 1.75), dpi=240)
GAL = {}
for ax, g in zip(axs, [1, 10, 100, 1000]):
    m = GraphSSL('propagation', gamma=g).fit(X, yy)
    P = m.predict_proba(G)[:, 1].reshape(gx.shape)
    acc = float((m.transduction_ == y).mean()); GAL[g] = acc
    ax.contourf(gx, gy, P, levels=[-1, 0.5, 2], colors=['#E3EEFB', '#FDF0E0'], zorder=0)
    ax.contour(gx, gy, P, levels=[0.5], colors=[GREEN], linewidths=1.3, zorder=3)
    for k in (0, 1):
        mm = unl & (y == k); ax.scatter(X[mm, 0], X[mm, 1], s=6, marker=MK[k], color=CL[k], lw=0, zorder=2)
    for i in lab: ax.scatter(X[i, 0], X[i, 1], s=44, marker=MK[y[i]], color=CL[y[i]], edgecolor=NAVY, lw=1.0, zorder=6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal'); ax.set_xlim(-1.5, 2.5); ax.set_ylim(-1.0, 1.5)
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.set_title(f'$\\gamma = {g}$', fontsize=9.5, color=NAVY, pad=2)
    ax.text(0.5, -0.03, f'accuracy {acc:.2f}', transform=ax.transAxes, ha='center', va='top', fontsize=9, color=GREY)
NUM['gallery_gamma_accuracy'] = {str(k): v for k, v in GAL.items()}
fig.subplots_adjust(wspace=0.04, left=0.005, right=0.995, top=0.9, bottom=0.08)
sv(fig, 'f392.png', 470)
print('gallery', GAL, round(time.time() - t0, 1))

# ---------------------------------------------------------------- f393 : the three knobs on the digits (30 labels, 5 draws)
Xd, yd = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.25, random_state=0, stratify=yd)
draws = []
for seed in range(5):
    idx, _ = train_test_split(np.arange(len(ytr)), train_size=30, random_state=seed, stratify=ytr)
    yl = -np.ones(len(ytr), int); yl[idx] = ytr[idx]; draws.append(yl)
def run(make):
    a = []
    for yl in draws:
        m = make().fit(Xtr, yl); a.append(m.score(Xte, yte))
    return np.mean(a), np.std(a)
R = {}
gam = [1e-4, 1e-3, 1e-2, 1e-1, 1, 3, 20]
R['gamma'] = [run(lambda g=g: LabelSpreading(gamma=g, alpha=0.8, max_iter=200)) for g in gam]
print('gamma', R['gamma'], round(time.time() - t0, 1))
ks = [3, 4, 5, 7, 10, 15, 20, 30]
R['k_LP'] = [run(lambda k=k: LabelPropagation(kernel='knn', n_neighbors=k, max_iter=1000)) for k in ks]
R['k_LS'] = [run(lambda k=k: LabelSpreading(kernel='knn', n_neighbors=k, alpha=0.8, max_iter=200)) for k in ks]
print('k', round(time.time() - t0, 1))
als = [0.05, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]
R['alpha_30'] = [run(lambda a=a: LabelSpreading(kernel='knn', n_neighbors=7, alpha=a, max_iter=30)) for a in als]
R['alpha_1000'] = [run(lambda a=a: LabelSpreading(kernel='knn', n_neighbors=7, alpha=a, max_iter=1000)) for a in als]
print('alpha', round(time.time() - t0, 1))
NUM['curves'] = {'gamma': gam, 'k': ks, 'alpha': als, **{k: [[round(a, 4), round(b, 4)] for a, b in v] for k, v in R.items()}}

fig, axs = plt.subplots(1, 3, figsize=(5.7, 2.15), dpi=240, gridspec_kw={'width_ratios': [1.25, 1, 1]})
def band(ax, xs, v, c, ls, lab_, mk='o', lw=1.4):
    m = np.array([a for a, b in v]); s = np.array([b for a, b in v])
    ax.fill_between(xs, m - s, m + s, color=c, alpha=0.13, lw=0)
    ax.plot(xs, m, color=c, ls=ls, lw=lw, marker=mk, ms=3.4, label=lab_)
ax = axs[0]; band(ax, np.arange(len(gam)), R['gamma'], PUR, '-', None)
ax.set_xticks(np.arange(len(gam))); ax.set_xticklabels(['$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '1', '3', '20'], fontsize=8.6)
ax.set_xlabel('$\\gamma$  (rbf)'); ax.set_ylabel('test accuracy'); ax.set_ylim(0, 1); ax.axvline(6, color=RED, lw=0.8, ls=':'); ax.text(5.75, 0.62, 'default', rotation=90, fontsize=8.8, color=RED, ha='right', va='center')
ax = axs[1]; xs = np.arange(len(ks))
band(ax, xs, R['k_LP'], BLUE, '-', 'LabelPropagation'); band(ax, xs, R['k_LS'], GREEN, '--', 'LabelSpreading', mk='s')
ax.set_xticks(xs); ax.set_xticklabels([str(k) for k in ks], fontsize=8.6); ax.set_xlabel('n_neighbors'); ax.set_ylim(0, 1)
ax.legend(loc='center right', fontsize=8.6, bbox_to_anchor=(1.0, 0.42), handlelength=1.6, labelspacing=0.2, borderpad=0.1)
ax = axs[2]; xs = np.arange(len(als))
band(ax, xs, R['alpha_1000'], GREEN, '-', '`max_iter` 1000'.replace('`', ''))
band(ax, xs, R['alpha_30'], RED, '--', '`max_iter` 30'.replace('`', ''), mk='s')
ax.set_xticks(xs); ax.set_xticklabels(['.05', '.2', '.4', '.6', '.8', '.9', '.95', '.99'], fontsize=8.4); ax.set_xlabel('$\\alpha$  (spreading)'); ax.set_ylim(0, 1)
ax.legend(loc='center left', fontsize=8.6, handlelength=1.6, labelspacing=0.2, borderpad=0.1)
for ax in axs: ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
for ax in axs[1:]: ax.set_yticklabels([])
fig.tight_layout(w_pad=0.5)
sv(fig, 'f393.png', 470)
json.dump(NUM, open('verify/fig_numbers2.json', 'w'), indent=1); json.dump(CHK, open('verify/fig_textsize2.json', 'w'), indent=1)
print(json.dumps(CHK)); print(round(time.time() - t0, 1), 's')
