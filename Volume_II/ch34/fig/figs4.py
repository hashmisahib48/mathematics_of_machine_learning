# Figures of Chapter 34, part 4: learning curves on the digits. Run: python3 fig/figs4.py
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

# ---------------------------------------------------------------- f3111 : learning curves on the digits
Xd, yd = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.25, random_state=0, stratify=yd)
lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
M = {'logistic, labelled only': (lr, False, '-', 'o', GREY), '1-NN, labelled only': (lambda: KNeighborsClassifier(1), False, '--', 's', GREY),
     'LabelPropagation knn 7': (lambda: LabelPropagation(kernel='knn', max_iter=1000), True, '-', 'o', BLUE),
     'LabelSpreading knn 7': (lambda: LabelSpreading(kernel='knn', alpha=0.8), True, '--', 's', GREEN),
     'LabelSpreading rbf 0.02': (lambda: LabelSpreading(gamma=0.02, alpha=0.8), True, '-.', '^', TEAL),
     'SelfTraining logistic': (lambda: SelfTrainingClassifier(lr()), True, ':', 'D', PUR)}
sizes = [10, 20, 30, 50, 100, 200, 400]; NS = 5
R = {k: np.zeros((len(sizes), NS)) for k in M}; T_ = {k: 0.0 for k in M}
for a, nl in enumerate(sizes):
    for seed in range(NS):
        idx, _ = train_test_split(np.arange(len(ytr)), train_size=nl, random_state=seed, stratify=ytr)
        yl = -np.ones(len(ytr), int); yl[idx] = ytr[idx]
        for k, (mk, semi, *_r) in M.items():
            t1 = time.time(); m = mk()
            m.fit(Xtr, yl) if semi else m.fit(Xtr[idx], ytr[idx])
            R[k][a, seed] = m.score(Xte, yte); T_[k] += time.time() - t1
    print(nl, round(time.time() - t0, 1), flush=True)
NUM['learning'] = {'sizes': sizes, 'draws': NS, 'mean': {k: R[k].mean(1).round(4).tolist() for k in M}, 'sd': {k: R[k].std(1).round(4).tolist() for k in M}, 'seconds': {k: round(v, 1) for k, v in T_.items()}}
fig, ax = plt.subplots(figsize=(4.6, 2.9), dpi=240)
for k, (mk, semi, ls, mrk, col) in M.items():
    ax.plot(sizes, R[k].mean(1), ls=ls, marker=mrk, ms=3.6, color=col, lw=1.4, label=k)
ax.set_xscale('log'); ax.set_xticks(sizes); ax.set_xticklabels([str(s_) for s_ in sizes]); ax.minorticks_off()
ax.set_xlabel('number of labelled training digits'); ax.set_ylabel('test accuracy'); ax.set_ylim(0.3, 1.0); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(loc='lower right', fontsize=8.6, handlelength=2.2, labelspacing=0.2, borderpad=0.1)
fig.tight_layout()
sv(fig, 'f3111.png', 420)
json.dump(NUM, open('verify/fig_numbers4.json', 'w'), indent=1); json.dump(CHK, open('verify/fig_textsize4.json', 'w'), indent=1)
print(json.dumps(CHK)); print(round(time.time() - t0, 1), 's')
