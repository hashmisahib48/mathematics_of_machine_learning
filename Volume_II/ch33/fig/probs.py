# Problem figures and answers of Chapter 33. Run: python3 fig/probs.py  (answers -> verify/problem_answers.json)
import sys, os, json, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../lib'); os.chdir(HERE + '/..')
from figstyle import *
import matplotlib.text as mtext
from matplotlib.patches import Rectangle
from sklearn.datasets import load_iris, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hamming_loss, accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
plt.rcParams.update({'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.labelsize': 9.5, 'legend.fontsize': 9})
A = {}; CHK = {}
CL = [BLUE, OR, TEAL]; MK = ['o', 's', '^']

def psv(fig, name, wpx=135):
    from PIL import Image
    fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig, name); iw, ih = Image.open(name).size
    printed = min(round(wpx * 1.55), round(235 * iw / ih)); w_in = iw / 240
    CHK[name] = {'fig_in': round(w_in, 2), 'printed_px': printed, 'min_font_design': round(fs, 1), 'min_font_printed': round(fs * printed / 96 / w_in, 1)}

def small(w=2.5, h=1.75): return plt.subplots(figsize=(w, h), dpi=240)

# ---------------------------------------------------------------- 33A.1 counting problems and rows
sizes = [400, 300, 200, 100, 50, 50]; K = 6; n = sum(sizes)
pairs = [(j, k) for j in range(K) for k in range(j + 1, K)]
rows_ovo = sum(sizes[j] + sizes[k] for j, k in pairs)
A['33A.1'] = {'n': n, 'K': K, 'ovr_problems': K, 'ovo_problems': len(pairs), 'ovr_rows': K * n, 'ovo_rows': rows_ovo, 'ovo_rows_formula': (K - 1) * n,
              'ovr_cost_sq': K * n ** 2, 'ovo_cost_sq': sum((sizes[j] + sizes[k]) ** 2 for j, k in pairs)}
fig, ax = small(2.5, 2.0)
for j in range(K):
    for k in range(K):
        if k > j:
            ax.add_patch(Rectangle((k, K - 1 - j), 1, 1, fc='#E3F4EA', ec='white', lw=1.5)); ax.text(k + .5, K - 1 - j + .5, str(sizes[j] + sizes[k]), ha='center', va='center', fontsize=8.5, color=NAVY)
        elif k == j:
            ax.add_patch(Rectangle((k, K - 1 - j), 1, 1, fc=LGREY, ec='white', lw=1.5)); ax.text(k + .5, K - 1 - j + .5, str(sizes[j]), ha='center', va='center', fontsize=8.5, color=BLUE, weight='bold')
for k in range(K): ax.text(k + .5, K + 0.12, f'{k + 1}', ha='center', fontsize=9, color=GREY); ax.text(-0.15, K - 1 - k + .5, f'{k + 1}', ha='right', va='center', fontsize=9, color=GREY)
ax.set_xlim(-0.5, K); ax.set_ylim(0, K + 0.6); ax.axis('off'); ax.set_aspect('equal'); psv(fig, 'p_33A1.png')

# ---------------------------------------------------------------- 33A.2 a tie in the votes
f01, f12, f02 = 0.8, 0.3, -0.6
f = {(0, 1): f01, (0, 2): f02, (1, 2): f12}
votes = np.zeros(3); c = np.zeros(3)
for (j, k), v in f.items():
    votes[k if v > 0 else j] += 1; c[j] -= v; c[k] += v
Fk = votes + c / (3 * (np.abs(c) + 1))
A['33A.2'] = {'votes': votes.tolist(), 'c': c.round(4).tolist(), 'F': Fk.round(4).tolist(), 'pred': int(Fk.argmax())}
fig, ax = new(2.5, 2.0)
P = {0: (0.0, 0.0), 1: (2.0, 0.0), 2: (1.0, 1.65)}
for k, (px, py) in P.items(): ax.add_patch(plt.Circle((px, py), 0.28, fc=['#E3EEFB', '#FDF0E0', '#DDF0F1'][k], ec=CL[k], lw=1.4)); ax.text(px, py, str(k), ha='center', va='center', fontsize=10, color=NAVY, weight='bold')
def edge(a, b, t, off):
    (x1, y1), (x2, y2) = P[a], P[b]; d = np.array([x2 - x1, y2 - y1]); d /= np.linalg.norm(d)
    arr(ax, (x1 + 0.3 * d[0], y1 + 0.3 * d[1]), (x2 - 0.3 * d[0], y2 - 0.3 * d[1]), c=NAVY, lw=1.0, ms=8)
    ax.text((x1 + x2) / 2 + off[0], (y1 + y2) / 2 + off[1], t, ha='center', va='center', fontsize=9, color=NAVY, bbox=dict(fc='white', ec='none', pad=1))
edge(0, 1, '$f_{01}$ = 0.8', (0, -0.2)); edge(1, 2, '$f_{12}$ = 0.3', (0.38, 0.0)); edge(0, 2, '$f_{02}$ = −0.6', (-0.38, 0.0))
ax.set_xlim(-0.8, 2.8); ax.set_ylim(-0.5, 2.1); psv(fig, 'p_33A2.png')

# ---------------------------------------------------------------- 33A.3 decoding a code word
M = np.array([[1, 1, 1, 1, 1, 1], [1, 1, -1, -1, -1, -1], [-1, -1, 1, 1, -1, -1], [-1, -1, -1, -1, 1, 1]])
s = np.array([-0.9, 0.5, 0.8, 0.7, -0.3, 0.6])
d = np.sqrt(((s - M) ** 2).sum(1)); ham = (M[:, None, :] != M[None, :, :]).sum(2); sg = np.sign(s).astype(int)
A['33A.3'] = {'dist': d.round(3).tolist(), 'nearest': 'ABCD'[int(d.argmin())], 'sign_disagreements': [int((sg != M[k]).sum()) for k in range(4)], 'min_hamming': int(ham[np.triu_indices(4, 1)].min()),
              'correctable': (int(ham[np.triu_indices(4, 1)].min()) - 1) // 2}
fig, ax = new(2.5, 1.9)
for i in range(4):
    for j in range(6):
        v = M[i, j]; ax.add_patch(Rectangle((j, 4 - i), 1, 1, fc='#FDF0E0' if v > 0 else LB, ec='white', lw=1.5)); ax.text(j + .5, 4.5 - i, '+' if v > 0 else '−', ha='center', va='center', fontsize=9.5, color=OR if v > 0 else BLUE, weight='bold')
    ax.text(-0.2, 4.5 - i, 'ABCD'[i], ha='right', va='center', fontsize=9.5, color=NAVY, weight='bold')
for j in range(6):
    ax.add_patch(Rectangle((j, 0.1), 1, 0.8, fc=LGREY, ec='white', lw=1.5)); ax.text(j + .5, 0.5, f'{s[j]:g}'.replace('-', '−'), ha='center', va='center', fontsize=8.5, color=PUR, weight='bold')
ax.text(-0.2, 0.5, '$\\mathbf{s}$', ha='right', va='center', fontsize=10, color=PUR)
ax.set_xlim(-0.8, 6.1); ax.set_ylim(0, 5.1); ax.axis('off'); ax.set_aspect('equal'); psv(fig, 'p_33A3.png')

# ---------------------------------------------------------------- 33A.4 multilabel scores
Y = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0]]); Yh = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 1], [0, 0, 0]])
A['33A.4'] = {'hamming': round(float(hamming_loss(Y, Yh)), 4), 'subset': round(float(accuracy_score(Y, Yh)), 4), 'f1_per_label': [round(float(v), 4) for v in f1_score(Y, Yh, average=None)],
              'f1_micro': round(float(f1_score(Y, Yh, average='micro')), 4), 'errors': int((Y != Yh).sum())}
fig, axs = plt.subplots(1, 2, figsize=(2.6, 1.7), dpi=240)
for ax, Mx, t in zip(axs, [Y, Yh], ['true  $Y$', 'predicted  $\\hat Y$']):
    for i in range(5):
        for j in range(3):
            bad = (Y[i, j] != Yh[i, j]) and Mx is Yh
            ax.add_patch(Rectangle((j, 4 - i), 1, 1, fc=PINK if bad else ('#FDF0E0' if Mx[i, j] else LGREY), ec='white', lw=1.3)); ax.text(j + .5, 4.5 - i, str(Mx[i, j]), ha='center', va='center', fontsize=9.5, color=RED if bad else NAVY, weight='bold' if bad else 'normal')
    ax.set_xlim(0, 3); ax.set_ylim(0, 5); ax.axis('off'); ax.set_title(t, fontsize=9.5, color=NAVY, pad=2)
fig.subplots_adjust(wspace=0.15, left=0.02, right=0.98, top=0.88, bottom=0.02); psv(fig, 'p_33A4.png')

# ---------------------------------------------------------------- 33B.1 normalised probabilities
X, y = load_iris(return_X_y=True); Z = StandardScaler().fit_transform(X)
ovr = OneVsRestClassifier(LogisticRegression(max_iter=300)).fit(Z, y); nat = LogisticRegression(max_iter=300).fit(Z, y)
i = 100
raw = np.array([e.predict_proba(Z[i:i + 1])[0, 1] for e in ovr.estimators_]); normed = raw / raw.sum()
A['33B.1'] = {'flower': i, 'raw': raw.round(4).tolist(), 'sum': round(float(raw.sum()), 4), 'normalised': normed.round(4).tolist(), 'predict_proba_ovr': ovr.predict_proba(Z[i:i + 1])[0].round(4).tolist(),
              'diff': float(np.abs(normed - ovr.predict_proba(Z[i:i + 1])[0]).max()), 'native': nat.predict_proba(Z[i:i + 1])[0].round(4).tolist(),
              'acc_ovr': round(float(ovr.score(Z, y)), 4), 'acc_native': round(float(nat.score(Z, y)), 4)}
fig, ax = small(2.4, 1.4); xx = np.arange(3); w = 0.27
ax.bar(xx - w, raw, w, color=LB, ec=BLUE, lw=0.8, label='raw $p_k$'); ax.bar(xx, normed, w, color='#E3F4EA', ec=GREEN, lw=0.8, label='normalised'); ax.bar(xx + w, nat.predict_proba(Z[i:i + 1])[0], w, color='#FDF0E0', ec=OR, lw=0.8, label='softmax')
ax.set_xticks(xx); ax.set_xticklabels(['class 0', 'class 1', 'class 2'], fontsize=9); ax.set_ylim(0, 1.15); ax.set_ylabel('probability'); ax.legend(loc='upper left', fontsize=8.5, ncol=1, handlelength=1, borderpad=0.1, labelspacing=0.15); psv(fig, 'p_33B1.png')

# ---------------------------------------------------------------- 33B.2 a tie in the votes
Xb, yb = make_blobs(n_samples=90, centers=3, cluster_std=2.5, random_state=1)
a = SVC(kernel='linear').fit(Xb, yb); b = OneVsOneClassifier(SVC(kernel='linear')).fit(Xb, yb); c_ = SVC(kernel='linear', break_ties=True).fit(Xb, yb)
G = np.random.default_rng(0).uniform(Xb.min(0) - 5, Xb.max(0) + 5, (200000, 2)); pa, pb, pc = a.predict(G), b.predict(G), c_.predict(G); dif = np.where(pa != pb)[0]
A['33B.2'] = {'differ': int(len(dif)), 'wrapper_vs_break_ties': int((pb != pc).sum()), 'example_point': G[dif[0]].round(3).tolist(), 'svc_pred': int(pa[dif[0]]), 'wrapper_pred': int(pb[dif[0]]),
              'scores_example': b.decision_function(G[dif[0]:dif[0] + 1])[0].round(4).tolist()}
fig, ax = small(2.6, 1.45)
for k in range(3): ax.scatter(Xb[yb == k, 0], Xb[yb == k, 1], s=7, color=CL[k], marker=MK[k], lw=0, zorder=3)
ax.scatter(G[dif, 0], G[dif, 1], s=22, facecolor='none', edgecolor=RED, lw=1.0, zorder=5)
ax.text(0.02, 0.97, 'red rings: SVC and\nthe wrapper differ', transform=ax.transAxes, va='top', fontsize=8.5, color=RED)
ax.set_xticks([]); ax.set_yticks([]); psv(fig, 'p_33B2.png')

# ---------------------------------------------------------------- 33B.3 a chain by hand
def make(n, rng):
    Xx = rng.uniform(-2, 2, (n, 2)); out = Xx[:, 0] > 0; beach = out & (Xx[:, 1] > 0); return Xx, np.c_[out, beach].astype(int)
rng = np.random.default_rng(0); Xtr, Ytr = make(300, rng); Xte, Yte = make(2000, rng)
base = LogisticRegression(C=1e4, max_iter=1000)
ch = ClassifierChain(base, order=[0, 1]).fit(Xtr, Ytr)
l1 = LogisticRegression(C=1e4, max_iter=1000).fit(Xtr, Ytr[:, 0]); l2 = LogisticRegression(C=1e4, max_iter=1000).fit(np.c_[Xtr, Ytr[:, 0]], Ytr[:, 1])
p1 = l1.predict(Xte); p2 = l2.predict(np.c_[Xte, p1]); Ph = np.c_[p1, p2]
ind = MultiOutputClassifier(base).fit(Xtr, Ytr)
A['33B.3'] = {'identical': bool((Ph == ch.predict(Xte)).all()), 'subset_chain': round(float(accuracy_score(Yte, Ph)), 4), 'subset_indep': round(float(accuracy_score(Yte, ind.predict(Xte))), 4),
              'beach_acc_chain': round(float((Ph[:, 1] == Yte[:, 1]).mean()), 4), 'beach_acc_indep': round(float((ind.predict(Xte)[:, 1] == Yte[:, 1]).mean()), 4),
              'link2_coef': l2.coef_.round(1).tolist()}
fig, ax = new(2.5, 1.5)
def bx_(xc, yc, w, h, t, fc, ec): box(ax, xc - w / 2, yc - h / 2, w, h, t, fc=fc, ec=ec, fs=9, r=0.05)
bx_(0.35, 0.75, 0.6, 0.45, '$\\mathbf{x}$', LGREY, GREY); bx_(1.55, 1.15, 0.8, 0.45, 'link 1', LB, BLUE); bx_(3.05, 0.55, 1.0, 0.45, 'link 2', LB, BLUE); bx_(4.35, 0.55, 0.7, 0.45, '$\\hat y_2$', '#E3F4EA', GREEN)
bx_(3.05, 1.35, 0.8, 0.4, '$\\hat y_1$', '#E3F4EA', GREEN)
arr(ax, (0.65, 0.88), (1.15, 1.1), c=NAVY, lw=0.9, ms=7); arr(ax, (1.95, 1.18), (2.65, 1.3), c=NAVY, lw=0.9, ms=7); arr(ax, (0.65, 0.65), (2.55, 0.55), c=NAVY, lw=0.9, ms=7); arr(ax, (3.05, 1.12), (3.05, 0.8), c=NAVY, lw=0.9, ms=7); arr(ax, (3.55, 0.55), (4.0, 0.55), c=NAVY, lw=0.9, ms=7)
ax.set_xlim(-0.1, 4.85); ax.set_ylim(0.1, 1.65); psv(fig, 'p_33B3.png')

# ---------------------------------------------------------------- 33C.1 masking
x = np.array([1., 2, 4, 5, 7, 8]); yy = np.array([0, 0, 1, 1, 2, 2])
fig, ax = small(2.6, 1.4); xs = np.linspace(0.5, 8.5, 50)
for k in range(3):
    zz = (yy == k) * 1.0; w = ((x - x.mean()) * (zz - zz.mean())).sum() / ((x - x.mean()) ** 2).sum(); ax.plot(xs, zz.mean() + w * (xs - x.mean()), color=CL[k], lw=1.5, ls=['-', '--', ':'][k])
    ax.scatter(x[yy == k], np.full(2, -0.1), s=26, color=CL[k], marker=MK[k], edgecolor='white', lw=0.6, zorder=5, clip_on=False)
ax.set_ylim(-0.17, 1.0); ax.set_xlim(0.5, 8.5); ax.set_xlabel('$x$'); ax.set_ylabel('score'); ax.grid(color=GRID, lw=0.5); psv(fig, 'p_33C1.png')

# ---------------------------------------------------------------- 33C.2 a chain of ridge regressors
fig, ax = new(2.5, 1.45)
bx_(0.35, 0.75, 0.6, 0.45, '$\\mathbf{x}$', LGREY, GREY); bx_(1.65, 0.75, 0.9, 0.45, 'ridge 1', LB, BLUE); bx_(3.15, 0.75, 0.9, 0.45, 'ridge 2', LB, BLUE); bx_(4.45, 0.75, 0.7, 0.45, '$\\hat y_2$', '#E3F4EA', GREEN)
arr(ax, (0.65, 0.75), (1.2, 0.75), c=NAVY, lw=0.9, ms=7); arr(ax, (2.1, 0.75), (2.7, 0.75), c=NAVY, lw=0.9, ms=7); arr(ax, (3.6, 0.75), (4.1, 0.75), c=NAVY, lw=0.9, ms=7)
ax.annotate('', xy=(3.15, 0.5), xytext=(0.35, 0.5), arrowprops=dict(arrowstyle='-|>', color=GREY, lw=0.8, connectionstyle='arc3,rad=0.25', mutation_scale=7))
ax.text(1.65, 1.2, '$\\hat y_1$', ha='center', fontsize=9.5, color=GREEN); ax.text(2.4, 1.2, 'enters link 2', ha='left', fontsize=9, color=GREY)
ax.set_xlim(-0.1, 4.9); ax.set_ylim(0.0, 1.5); psv(fig, 'p_33C2.png')

# ---------------------------------------------------------------- 33C.3 five thousand classes
Kc = 5000; nprob = {'OvR': Kc, 'OvO': Kc * (Kc - 1) // 2, 'ECOC 1.5': int(Kc * 1.5)}
A['33C.3'] = nprob
fig, ax = small(2.4, 1.4); names = list(nprob); vals = list(nprob.values())
ax.barh(names[::-1], vals[::-1], color=[OR, BLUE, BLUE][::-1] if False else [PUR, BLUE, OR], height=0.55); ax.set_xscale('log'); ax.set_xlim(1e3, 2e7)
for yi, v in zip([2, 1, 0], vals):
    s_ = f'{v:,}'.replace(',', ' ')
    (ax.text(v * 0.8, yi, s_, va='center', ha='right', fontsize=8.8, color='white') if v > 1e6 else ax.text(v * 1.25, yi, s_, va='center', fontsize=8.8, color=NAVY))
ax.set_xlabel('binary problems  (log scale)'); ax.grid(axis='x', color=GRID, lw=0.5); ax.set_axisbelow(True); psv(fig, 'p_33C3.png')

json.dump(A, open('verify/problem_answers.json', 'w'), indent=1); json.dump(CHK, open('verify/probfig_textsize.json', 'w'), indent=1)
print(json.dumps(A, indent=0)); print(json.dumps(CHK))
