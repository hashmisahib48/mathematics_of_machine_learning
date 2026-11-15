# Figures of Chapter 33, part 2: experiments. Run (after code/case_digits.py): python3 fig/figs2.py
import sys, os, json, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../lib'); os.chdir(HERE + '/..')
from figstyle import *
import matplotlib.text as mtext
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OutputCodeClassifier, OneVsRestClassifier, OneVsOneClassifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
plt.rcParams.update({'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.labelsize': 9.5, 'legend.fontsize': 9})
NUM = {}; CHK = {}
CL5 = [BLUE, OR, TEAL, PUR, '#6B7280']; MK5 = ['o', 's', '^', 'D', 'v']
LIGHT5 = ['#DCE8F8', '#FCE9CF', '#D3ECEE', '#E6DCF0', '#E4E6EA']

def sv(fig, name, printed_px):
    fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig, name)
    from PIL import Image
    w = Image.open(name).size[0] / 240
    CHK[name] = {'fig_in': round(w, 2), 'printed_in': round(printed_px / 96, 2), 'min_font_design': round(fs, 1),
                 'min_font_printed': round(fs * printed_px / 96 / w, 1)}

# ---------------------------------------------------------------- f391 : gallery over code_size
cen = [[0, 0], [2.5, 0], [1.2, 2.2], [-1.3, 2.2], [3.5, 2.4]]
X, y = make_blobs(n_samples=250, centers=cen, cluster_std=0.55, random_state=3)
Xt, yt = make_blobs(n_samples=1000, centers=cen, cluster_std=0.55, random_state=4)
g = np.mgrid[-2.4:5.4:220j, -1.5:4.0:220j]; G = np.c_[g[0].ravel(), g[1].ravel()]
fig, axs = plt.subplots(1, 4, figsize=(6.2, 1.95), dpi=240, sharey=True, gridspec_kw={'wspace': 0.06})
NUM['gallery'] = []
for ax, cs in zip(axs, [0.6, 1.0, 2.0, 8.0]):
    m = OutputCodeClassifier(LogisticRegression(), code_size=cs, random_state=1).fit(X, y)
    Z = m.predict(G).reshape(220, 220); acc = m.score(Xt, yt)
    ax.contourf(g[0], g[1], Z, levels=np.arange(-0.5, 5), colors=LIGHT5, zorder=0)
    for k in range(5):
        s_ = y == k; ax.scatter(X[s_, 0], X[s_, 1], s=7, color=CL5[k], marker=MK5[k], lw=0, zorder=3)
    ax.set_title(f'code_size {cs:g}\n{len(m.estimators_)} bits, test {acc:.2f}', fontsize=9.5, color=NAVY, pad=3)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-2.4, 5.4); ax.set_ylim(-1.5, 4.0); ax.set_aspect('equal')
    for sp_ in ax.spines.values(): sp_.set_visible(True); sp_.set_color(GRID)
    NUM['gallery'].append({'code_size': cs, 'bits': len(m.estimators_), 'test_acc': round(float(acc), 3)})
sv(fig, 'f391.png', 470)

# ---------------------------------------------------------------- f392 : validation curve on the digits
Xd, yd = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.25, random_state=0, stratify=yd)
sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
base = LogisticRegression(max_iter=300)
ovr_acc = OneVsRestClassifier(base).fit(Xtr, ytr).score(Xte, yte); ovo_acc = OneVsOneClassifier(base).fit(Xtr, ytr).score(Xte, yte)
CS = [0.5, 1, 1.5, 2, 3, 5]; SEEDS = range(4); acc = np.zeros((len(CS), len(SEEDS))); ham = np.zeros_like(acc); bits = []
for i, cs in enumerate(CS):
    for j, rs in enumerate(SEEDS):
        m = OutputCodeClassifier(base, code_size=cs, random_state=rs).fit(Xtr, ytr)
        M = m.code_book_; h = (M[:, None, :] != M[None, :, :]).sum(2)[np.triu_indices(10, 1)].min()
        acc[i, j] = m.score(Xte, yte); ham[i, j] = h
    bits.append(len(m.estimators_))
NUM['valcurve'] = {'code_size': CS, 'bits': bits, 'acc_mean': acc.mean(1).round(4).tolist(), 'acc_min': acc.min(1).round(4).tolist(), 'acc_max': acc.max(1).round(4).tolist(),
                   'minham_mean': ham.mean(1).round(2).tolist(), 'ovr': round(float(ovr_acc), 4), 'ovo': round(float(ovo_acc), 4)}
fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.35), dpi=240, gridspec_kw={'wspace': 0.34})
ax = axs[0]
ax.fill_between(CS, acc.min(1), acc.max(1), color='#E6DCF0', zorder=1, label='range, 4 code books')
ax.plot(CS, acc.mean(1), '-o', color=PUR, lw=1.8, ms=4.5, zorder=3, label='output code, mean')
ax.axhline(ovo_acc, color=GREEN, lw=1.3, ls='--', label='one-versus-one'); ax.axhline(ovr_acc, color=BLUE, lw=1.3, ls=':', label='one-versus-rest')
ax.set_xlabel('code_size'); ax.set_ylabel('test accuracy'); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_ylim(0.5, 1.0)
ax.legend(loc='lower right', fontsize=8.5, handlelength=1.6, labelspacing=0.3, frameon=True, facecolor='white', framealpha=0.95, edgecolor='none', borderpad=0.3)
ax = axs[1]
ax.plot(bits, ham.mean(1), '-s', color=RED, lw=1.8, ms=4.5); ax.fill_between(bits, ham.min(1), ham.max(1), color=PINK, zorder=0)
ax.set_xlabel('number of bits  $B$'); ax.set_ylabel('smallest Hamming distance'); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.subplots_adjust(left=0.1, right=0.985, bottom=0.2, top=0.96)
sv(fig, 'f392.png', 470)

# ---------------------------------------------------------------- f3101 : why a chain helps
def make(n, rng):
    Xx = rng.uniform(-2, 2, (n, 2)); out = Xx[:, 0] > 0; beach = out & (Xx[:, 1] > 0); return Xx, np.c_[out, beach].astype(int)
rng = np.random.default_rng(0); Xa, Ya = make(300, rng)
b = LogisticRegression(C=1e4, max_iter=1000)
ind = MultiOutputClassifier(b).fit(Xa, Ya); chn = ClassifierChain(b, order=[0, 1]).fit(Xa, Ya)
g = np.mgrid[-2:2:240j, -2:2:240j]; G = np.c_[g[0].ravel(), g[1].ravel()]
fig, axs = plt.subplots(1, 3, figsize=(5.6, 2.15), dpi=240, sharey=True, gridspec_kw={'wspace': 0.08})
grp = Ya[:, 0] + Ya[:, 1]  # 0 neither, 1 outdoor only, 2 beach
names = ['neither', 'outdoor only', 'beach']
for k, (nm, c_, mk) in enumerate(zip(names, [BLUE, OR, TEAL], ['o', 's', '^'])):
    axs[0].scatter(Xa[grp == k, 0], Xa[grp == k, 1], s=11, color=c_, marker=mk, lw=0, zorder=3, label=nm)
axs[0].legend(loc='upper center', bbox_to_anchor=(0.62, -0.2), ncol=3, fontsize=8.5, handletextpad=0.1, columnspacing=0.8, frameon=False, markerscale=1.1)
axs[0].set_title('the data', fontsize=9.5, color=NAVY, pad=3)
for ax, mdl, ttl in [(axs[1], ind, 'independent'), (axs[2], chn, 'chain')]:
    Z = mdl.predict(G)[:, 1].reshape(240, 240)
    ax.contourf(g[0], g[1], Z, levels=[-0.5, 0.5, 1.5], colors=['#E3EEFB', '#D3ECEE'], zorder=0)
    ax.contour(g[0], g[1], Z, levels=[0.5], colors=[GREEN], linewidths=1.6, zorder=2)
    ax.scatter(Xa[Ya[:, 1] == 0, 0], Xa[Ya[:, 1] == 0, 1], s=6, color=BLUE, lw=0, zorder=3); ax.scatter(Xa[Ya[:, 1] == 1, 0], Xa[Ya[:, 1] == 1, 1], s=8, color=TEAL, marker='^', lw=0, zorder=3)
    ax.set_title(ttl + ': region of beach', fontsize=9.5, color=NAVY, pad=3)
for ax in axs: ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal'); ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
for ax in axs: ax.set_xlabel('$x_1$ (outdoor if positive)', fontsize=9)
axs[0].set_ylabel('$x_2$', fontsize=9.5)
sv(fig, 'f3101.png', 470)

# ---------------------------------------------------------------- f3111 : digits case
rows = json.load(open('verify/case_digits.json'))
fig, axs = plt.subplots(1, 2, figsize=(5.3, 2.5), dpi=240, sharey=True, gridspec_kw={'wspace': 0.08, 'width_ratios': [1, 1]})
labs = [r[0].replace(' (softmax)', '\n(softmax)').replace(' (OvR inside)', '\n(OvR inside)').replace(' (OvO inside)', '\n(OvO inside)') for r in rows]
yy = np.arange(len(rows))[::-1]
cols = [GREEN if ('inside' in r[0] or 'softmax' in r[0]) else (BLUE if r[0].startswith('OvR') else OR) for r in rows]
axs[0].barh(yy, [r[4] for r in rows], color=cols, height=0.62); axs[0].set_xlim(0.94, 0.99)
for yi, r in zip(yy, rows): axs[0].text(r[4] + 0.0006, yi, f'{r[4]:.4f}', va='center', fontsize=9.2, color=NAVY)
axs[0].set_yticks(yy); axs[0].set_yticklabels(labs, fontsize=9.2); axs[0].set_xlabel('test accuracy'); axs[0].set_title('accuracy', fontsize=9.5, color=NAVY)
axs[1].barh(yy, [r[3] for r in rows], color=cols, height=0.62); axs[1].set_xscale('log'); axs[1].set_xlim(0.03, 3.2)
for yi, r in zip(yy, rows): axs[1].text(r[3] * 1.12, yi, f'{r[3]:.2f} s', va='center', fontsize=9.2, color=NAVY)
axs[1].set_xlabel('fit time (s, log scale)'); axs[1].set_title('time to fit', fontsize=9.5, color=NAVY)
for ax in axs: ax.grid(axis='x', color=GRID, lw=0.5); ax.set_axisbelow(True)
sv(fig, 'f3111.png', 470)

json.dump(NUM, open('verify/fig_numbers2.json', 'w'), indent=1); json.dump(CHK, open('verify/fig_textsize2.json', 'w'), indent=1)
print(json.dumps(NUM)); print(json.dumps(CHK))
