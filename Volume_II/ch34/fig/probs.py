# Problem figures and answers of Chapter 34. Run: python3 fig/probs.py  (answers -> verify/problem_answers.json)
import sys, os, json, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../lib'); os.chdir(HERE + '/..')
from figstyle import *
import matplotlib.text as mtext
from matplotlib.patches import Rectangle
from sklearn.datasets import load_digits, make_blobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier
warnings.simplefilter('ignore', ConvergenceWarning); warnings.simplefilter('ignore', RuntimeWarning)
plt.rcParams.update({'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.labelsize': 9.5, 'legend.fontsize': 9})
A = {}; CHK = {}
CL = [BLUE, OR]; MK = ['o', 's']

def psv(fig, name, wpx=135):
    from PIL import Image
    fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig, name); iw, ih = Image.open(name).size
    printed = min(round(wpx * 1.55), round(235 * iw / ih)); w_in = iw / 240
    CHK[name] = {'fig_in': round(w_in, 2), 'printed_px': printed, 'min_font_design': round(fs, 1), 'min_font_printed': round(fs * printed / 96 / w_in, 1)}

def small(w=2.5, h=1.75): return plt.subplots(figsize=(w, h), dpi=240)

def graph(ax, pos, edges, labels, kinds, arcs=(), ew=1.0, wfs=9.5, nfs=9.5, ns=260):
    """pos: node -> (x, y); edges: (i, j, weight); kinds: node -> None | 0 | 1 (labelled class)."""
    for (i, j, w_) in edges:
        (x0, y0), (x1, y1) = pos[i], pos[j]
        if (i, j) in arcs:
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-', color=GREY, lw=0.8 + ew * w_, connectionstyle='arc3,rad=-0.45'), zorder=1)
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.62, str(w_), ha='center', fontsize=wfs, color=NAVY)
        else:
            ax.plot([x0, x1], [y0, y1], color=GREY, lw=0.8 + ew * w_, zorder=1, solid_capstyle='butt')
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.2, str(w_), ha='center', fontsize=wfs, color=NAVY)
    for n, (x0, y0) in pos.items():
        k = kinds[n]
        if k is None: ax.scatter(x0, y0, s=ns, color='#E9EDF3', edgecolor=NAVY, lw=1.0, zorder=5)
        else: ax.scatter(x0, y0, s=ns, marker=MK[k], color=CL[k], edgecolor=NAVY, lw=1.1, zorder=5)
        ax.text(x0, y0, labels[n], ha='center', va='center', fontsize=nfs, color='white' if k is not None else NAVY, weight='bold', zorder=6)

# ---------------------------------------------------------------- 34A.1 a four-node graph (harmonic solution)
W = np.zeros((4, 4))                       # nodes A, u, v, B
for (i, j, w_) in [(0, 1, 3), (1, 2, 2), (2, 3, 4), (1, 3, 1)]: W[i, j] = W[j, i] = w_
d = W.sum(1); L, U = [0, 3], [1, 2]
fU = np.linalg.solve(np.diag(d[U]) - W[np.ix_(U, U)], W[np.ix_(U, L)] @ np.array([0., 1.]))
A['34A.1'] = {'f_u': float(fU[0]), 'f_v': float(fU[1]), 'fractions': '7/16, 13/16'}
fig, ax = small(2.6, 1.6); ax.axis('off'); ax.set_xlim(-0.5, 3.5); ax.set_ylim(-1.15, 0.6)
pos = {0: (0, 0), 1: (1, 0), 2: (2, -0.85), 3: (3, 0)}
for (i, j, w_, off) in [(0, 1, 3, (0, 0.22)), (1, 2, 2, (-0.2, 0.0)), (2, 3, 4, (0.2, 0.0)), (1, 3, 1, (0, 0.22))]:
    (x0, y0), (x1, y1) = pos[i], pos[j]
    ax.plot([x0, x1], [y0, y1], color=GREY, lw=0.8 + 0.6 * w_, zorder=1, solid_capstyle='butt')
    ax.text((x0 + x1) / 2 + off[0], (y0 + y1) / 2 + off[1], str(w_), ha='center', va='center', fontsize=9.5, color=NAVY)
for n, (lab_, k) in {0: ('A', 0), 1: ('u', None), 2: ('v', None), 3: ('B', 1)}.items():
    x0, y0 = pos[n]
    if k is None: ax.scatter(x0, y0, s=260, color='#E9EDF3', edgecolor=NAVY, lw=1.0, zorder=5)
    else: ax.scatter(x0, y0, s=260, marker=MK[k], color=CL[k], edgecolor=NAVY, lw=1.1, zorder=5)
    ax.text(x0, y0, lab_, ha='center', va='center', fontsize=9.5, color='white' if k is not None else NAVY, weight='bold', zorder=6)
psv(fig, 'p_34A1.png', 135)

# ---------------------------------------------------------------- 34A.2 label spreading, three nodes
W3 = np.array([[0, 1, 0], [1, 0, 3], [0, 3, 0.]]); d3 = W3.sum(1); S3 = W3 / np.sqrt(np.outer(d3, d3))
Y3 = np.array([[1, 0], [0, 0], [0, 1.]]); al = 0.5
F1 = al * S3 @ Y3 + (1 - al) * Y3
Fs = (1 - al) * np.linalg.solve(np.eye(3) - al * S3, Y3); Fn = Fs / Fs.sum(1, keepdims=True)
W3c = W3.copy(); X3 = np.arange(3.).reshape(-1, 1)
ls3 = LabelSpreading(kernel=lambda a, b: W3c.copy(), alpha=al, max_iter=100000, tol=1e-14).fit(X3, np.array([0, -1, 1]))
A['34A.2'] = {'S12': float(S3[0, 1]), 'S23': float(S3[1, 2]), 'F1_node2': F1[1].round(4).tolist(), 'F1': F1.round(4).tolist(), 'closed_node2': Fn[1].round(4).tolist(),
              'sklearn_node2': ls3.label_distributions_[1].round(4).tolist(), 'max_diff_sklearn': float(abs(ls3.label_distributions_ - Fn).max()), 'class_node2': int(Fn[1].argmax())}
fig, ax = small(2.4, 1.2); ax.axis('off'); ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.55, 0.7)
graph(ax, {0: (0, 0), 1: (1, 0), 2: (2, 0)}, [(0, 1, 1), (1, 2, 3)], {0: '1', 1: '2', 2: '3'}, {0: 0, 1: None, 2: 1}, ew=0.9)
psv(fig, 'p_34A2.png', 135)

# ---------------------------------------------------------------- 34A.3 a component without a label
W5 = np.zeros((5, 5))
for (i, j, w_) in [(0, 1, 2), (1, 2, 1), (3, 4, 3)]: W5[i, j] = W5[j, i] = w_
U5 = [1, 3, 4]; L5 = [0, 2]; d5 = W5.sum(1)
M5 = np.diag(d5[U5]) - W5[np.ix_(U5, U5)]
f2 = (W5[1, 0] * 0 + W5[1, 2] * 1) / d5[1]
lp5 = LabelPropagation(kernel=lambda a, b: W5.copy(), max_iter=100000, tol=1e-12).fit(np.arange(5.).reshape(-1, 1), np.array([0, -1, 1, -1, -1]))
A['34A.3'] = {'f_node2': float(f2), 'det_UU': float(np.linalg.det(M5)), 'det_block_45': float(np.linalg.det(M5[1:, 1:])), 'distributions': lp5.label_distributions_.round(4).tolist(), 'transduction': lp5.transduction_.tolist(), 'classes': lp5.classes_.tolist()}
fig, ax = small(2.5, 1.3); ax.axis('off'); ax.set_xlim(-0.5, 4.8); ax.set_ylim(-0.6, 0.75)
graph(ax, {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3.1, 0), 4: (4.1, 0)}, [(0, 1, 2), (1, 2, 1), (3, 4, 3)], {i: str(i + 1) for i in range(5)}, {0: 0, 1: None, 2: 1, 3: None, 4: None}, ew=0.9, ns=190, nfs=9, wfs=9)
psv(fig, 'p_34A3.png', 140)

# ---------------------------------------------------------------- 34A.4 a directed neighbour graph
x4 = np.array([0., 1, 3, 7]).reshape(-1, 1); y4 = np.array([0, -1, -1, 1])
lp4 = LabelPropagation(kernel='knn', n_neighbors=2, max_iter=100000, tol=1e-12).fit(x4, y4)
from sklearn.neighbors import NearestNeighbors
G4 = NearestNeighbors(n_neighbors=2).fit(x4).kneighbors_graph(x4, 2, mode='connectivity').toarray()
A['34A.4'] = {'W': G4.astype(int).tolist(), 'transduction': lp4.transduction_.tolist(), 'distributions': lp4.label_distributions_.round(4).tolist()}
fig, ax = small(2.6, 1.25); ax.axis('off'); ax.set_xlim(-0.3, 7.5); ax.set_ylim(-0.95, 1.3)
for k, xv in enumerate(x4[:, 0]):
    kk = {0: 0, 3: 1}.get(k)
    if kk is None: ax.scatter(xv, 0, s=170, color='#E9EDF3', edgecolor=NAVY, lw=1.0, zorder=5)
    else: ax.scatter(xv, 0, s=170, marker=MK[kk], color=CL[kk], edgecolor=NAVY, lw=1.1, zorder=5)
    ax.text(xv, 0, str(k + 1), ha='center', va='center', fontsize=9, color='white' if kk is not None else NAVY, weight='bold', zorder=6)
    ax.text(xv, -0.62, f'$x$={xv:.0f}', ha='center', fontsize=9, color=GREY)
for (i, j, st_) in [(1, 0, '<|-|>'), (2, 1, '-|>'), (3, 2, '-|>')]:
    ax.annotate('', xy=(x4[j, 0], 0.17), xytext=(x4[i, 0], 0.17), arrowprops=dict(arrowstyle=st_, color=NAVY, lw=0.9, connectionstyle='arc3,rad=-0.7' if i == 1 else 'arc3,rad=-0.55', mutation_scale=7, shrinkA=7, shrinkB=7), zorder=3)
psv(fig, 'p_34A4.png', 140)

# ---------------------------------------------------------------- 34B.1 iterations against tolerance (chain of Example 34.5-1)
Wc = np.zeros((5, 5))
for i, w_ in enumerate([2, 4, 1, 1]): Wc[i, i + 1] = Wc[i + 1, i] = w_
xc = np.arange(5.).reshape(-1, 1); yc = np.array([0, -1, -1, -1, 1]); exact = np.array([0, 2 / 11, 3 / 11, 7 / 11, 1])
tols = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-9, 1e-12]; rows = []
for t in tols:
    m = LabelPropagation(kernel=lambda a, b: Wc.copy(), max_iter=100000, tol=t).fit(xc, yc)
    rows.append({'tol': t, 'n_iter': int(m.n_iter_), 'max_err': float(abs(m.label_distributions_[:, 1] - exact).max())})
A['34B.1'] = {'rows': rows, 'rho': float(max(abs(np.linalg.eigvals((Wc / Wc.sum(1)[:, None])[np.ix_([1, 2, 3], [1, 2, 3])]))))}
fig, ax = small(2.5, 1.75)
ax.semilogy([r['tol'] for r in rows], [r['max_err'] for r in rows], marker='o', ms=3.5, color=PUR, lw=1.3)
ax.set_xscale('log'); ax.invert_xaxis(); ax.set_xlabel('tol'); ax.set_ylabel('largest error'); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.set_xticks([1e-1, 1e-4, 1e-8, 1e-12]); ax.minorticks_off()
psv(fig, 'p_34B1.png', 135)

# ---------------------------------------------------------------- 34B.2 transduction against predict
Xd, yd = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.25, random_state=0, stratify=yd)
lab, _ = train_test_split(np.arange(len(ytr)), train_size=30, random_state=0, stratify=ytr)
yl = -np.ones(len(ytr), int); yl[lab] = ytr[lab]; Um = yl == -1
R2 = {}
for name, m in [('LabelPropagation', LabelPropagation(kernel='knn', n_neighbors=7, max_iter=1000)), ('LabelSpreading', LabelSpreading(kernel='knn', n_neighbors=7, alpha=0.8))]:
    m.fit(Xtr, yl); p = m.predict(Xtr); dif = p != m.transduction_
    R2[name] = {'differ_total': int(dif.sum()), 'differ_labelled': int(dif[~Um].sum()), 'differ_unlabelled': int(dif[Um].sum()),
                'acc_transduction_unlabelled': float((m.transduction_[Um] == ytr[Um]).mean()), 'acc_predict_unlabelled': float((p[Um] == ytr[Um]).mean())}
A['34B.2'] = R2
fig, ax = small(2.5, 1.75)
names = list(R2); vals = [R2[n]['differ_total'] for n in names]
ax.bar(['propagation', 'spreading'], vals, color=[BLUE, GREEN], width=0.55)
for k, v in enumerate(vals): ax.text(k, v + 1.5, str(v), ha='center', fontsize=9.5, color=NAVY)
ax.set_ylabel('predictions that differ'); ax.set_ylim(0, max(vals) * 1.25); ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
psv(fig, 'p_34B2.png', 135)

# ---------------------------------------------------------------- 34B.3 thresholds of self-training
Xb, yb = make_blobs(400, centers=[[-1.5, 0], [1.5, 0]], cluster_std=1.6, random_state=5)
Xbt, ybt = make_blobs(4000, centers=[[-1.5, 0], [1.5, 0]], cluster_std=1.6, random_state=777)
rng = np.random.default_rng(5); idx = np.r_[rng.choice(np.where(yb == 0)[0], 5, replace=False), rng.choice(np.where(yb == 1)[0], 5, replace=False)]
yy = -np.ones(400, int); yy[idx] = yb[idx]
base = LogisticRegression().fit(Xb[idx], yb[idx]).score(Xbt, ybt)
rows = []
for thr in [0.6, 0.8, 0.9, 0.95, 0.99]:
    st = SelfTrainingClassifier(LogisticRegression(), threshold=thr, max_iter=None).fit(Xb, yy)
    rows.append({'threshold': thr, 'n_iter': int(st.n_iter_), 'n_pseudo': int((st.labeled_iter_ > 0).sum()), 'unlabelled_left': int((st.transduction_ == -1).sum()),
                 'termination': st.termination_condition_, 'test_acc': round(float(st.score(Xbt, ybt)), 4)})
A['34B.3'] = {'supervised_acc': round(float(base), 4), 'rows': rows}
fig, ax = small(2.5, 1.75)
ax.plot([r['threshold'] for r in rows], [r['n_pseudo'] for r in rows], marker='o', ms=3.5, color=PUR, lw=1.3)
ax.set_xlabel('threshold'); ax.set_ylabel('pseudo-labels added'); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
psv(fig, 'p_34B3.png', 135)

# ---------------------------------------------------------------- 34C.1 averages
fig, ax = small(2.5, 1.45); ax.axis('off'); ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.3, 1.25)
fv = [0, 2 / 11, 3 / 11, 7 / 11, 1]
ax.plot([0, 4], [0, 0], color=GRID, lw=0.8, zorder=1)
for i in range(5):
    k = {0: 0, 4: 1}.get(i)
    ax.vlines(i, 0, fv[i], color=GREEN, lw=2.2, zorder=2)
    ax.scatter(i, fv[i], s=36, color=GREEN if k is None else CL[k], marker='o' if k in (None, 0) else 's', edgecolor=NAVY, lw=0.9, zorder=5)
    ax.text(i, -0.09, str(i + 1), ha='center', va='top', fontsize=9.5, color=NAVY)
ax.text(2, 1.12, '$f_i$ lies between 0 and 1', ha='center', fontsize=9.5, color=GREY)
psv(fig, 'p_34C1.png', 135)

# ---------------------------------------------------------------- 34C.2 the random walk
fig, ax = small(2.6, 1.45); ax.axis('off'); ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.9, 1.0)
graph(ax, {i: (i, 0) for i in range(5)}, [(0, 1, 2), (1, 2, 4), (2, 3, 1), (3, 4, 1)], {i: str(i + 1) for i in range(5)}, {0: 0, 1: None, 2: None, 3: None, 4: 1}, ew=0.35, ns=210, nfs=9, wfs=9)
ax.annotate('', xy=(1.78, -0.42), xytext=(2.0, -0.2), arrowprops=dict(arrowstyle='-|>', color=RED, lw=1.1, mutation_scale=7))
ax.annotate('', xy=(2.22, -0.42), xytext=(2.0, -0.2), arrowprops=dict(arrowstyle='-|>', color=RED, lw=0.7, mutation_scale=7))
ax.text(1.45, -0.62, '4/5', fontsize=9, color=RED, ha='center'); ax.text(2.6, -0.62, '1/5', fontsize=9, color=RED, ha='center')
psv(fig, 'p_34C2.png', 135)

# ---------------------------------------------------------------- 34C.3 a convenience sample of labels
Xo, yo = make_blobs(300, centers=[[-1.5, 0], [1.5, 0]], cluster_std=1.6, random_state=4)
fig, ax = small(2.5, 1.4)
for k in (0, 1):
    m = yo == k; ax.scatter(Xo[m, 0], Xo[m, 1], s=5, marker=MK[k], color=CL[k], lw=0, alpha=0.45, zorder=2)
top = np.argsort(-Xo[:, 1])
pick = {0: [i for i in top if yo[i] == 0][:3], 1: [i for i in top if yo[i] == 1][:3]}
for k in (0, 1):
    for i in pick[k]: ax.scatter(Xo[i, 0], Xo[i, 1], s=40, marker=MK[k], color=CL[k], edgecolor=NAVY, lw=1.0, zorder=6)
ax.text(0.5, -0.02, 'labels: only the top of the cloud', transform=ax.transAxes, fontsize=9, color=NAVY, va='top', ha='center')
ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
for s_ in ax.spines.values(): s_.set_visible(False)
psv(fig, 'p_34C3.png', 135)

json.dump(A, open('verify/problem_answers.json', 'w'), indent=1); json.dump(CHK, open('verify/probfig_textsize.json', 'w'), indent=1)
print(json.dumps(A, indent=0)); print(json.dumps(CHK))
