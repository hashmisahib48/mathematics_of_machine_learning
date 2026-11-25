import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from figstyle import *
from matplotlib.patches import FancyBboxPatch, Rectangle
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
P = lambda n: os.path.join(HERE, 'png', n)
def fin(fig, n): fig.tight_layout(pad=0.3); fig.savefig(P(n), dpi=300, facecolor='white'); plt.close(fig)
def sty(ax, xl, yl): ax.set_xlabel(xl, fontsize=7.2); ax.set_ylabel(yl, fontsize=7.2); ax.tick_params(labelsize=6.8, length=2.2)
# 11.2-1 k-fold
fig, ax = plt.subplots(figsize=(2.05, 1.45), dpi=300)
for f in range(5):
    for g in range(5):
        ax.add_patch(Rectangle((g, 4 - f), 0.94, 0.8, fc=OR if g == f else '#DCE8F8', ec=NAVY, lw=0.4))
    ax.text(-0.15, 4 - f + 0.4, f'fold {f + 1}', ha='right', va='center', fontsize=6.6)
    ax.text(5.15, 4 - f + 0.4, f'$a_{f + 1}$', ha='left', va='center', fontsize=7)
ax.set_xlim(-1.3, 5.7); ax.set_ylim(-0.1, 5.2); ax.axis('off')
ax.text(2.5, 5.0, 'training (blue) and test (orange) parts', ha='center', fontsize=6.5, color=GREY); fin(fig, 'f111.png')
# 11.4-1 ROC example
y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]); s = np.array([0.9, 0.8, 0.55, 0.3, 0.7, 0.4, 0.35, 0.2, 0.1, 0.05])
fpr, tpr, th = roc_curve(y, s); fig, ax = plt.subplots(figsize=(2.05, 1.75), dpi=300)
ax.fill_between(fpr, 0, tpr, step='post', color=BLUE, alpha=0.15); ax.step(fpr, tpr, where='post', color=BLUE, lw=1.1)
ax.plot([0, 1], [0, 1], color=GREY, lw=0.5, ls='--'); ax.plot(1 / 6, 0.75, 'o', ms=4, color=RED); ax.text(0.2, 0.68, 'threshold 0.5', fontsize=6.6, color=RED)
ax.text(0.55, 0.3, 'AUC = 0.833', fontsize=6.8, color=BLUE); ax.set_aspect('equal'); sty(ax, 'false positive rate', 'true positive rate'); fin(fig, 'f112.png')
# 11.7-1 ROC curves of four models
X, Y = load_breast_cancer(return_X_y=True); a, b, c, d = train_test_split(X, Y, test_size=0.25, random_state=0, stratify=Y)
M = [('k-NN', make_pipeline(StandardScaler(), KNeighborsClassifier(7)), BLUE), ('logistic', make_pipeline(StandardScaler(), LogisticRegression()), RED),
     ('naive Bayes', GaussianNB(), GREEN), ('tree', DecisionTreeClassifier(max_depth=4, random_state=0), OR)]
fig, ax = plt.subplots(figsize=(2.05, 1.85), dpi=300)
for n, m, col in M:
    p = m.fit(a, c).predict_proba(b)[:, 1]; f, t, _ = roc_curve(d, p); ax.plot(f, t, color=col, lw=0.9, label=f'{n} ({roc_auc_score(d, p):.3f})')
ax.plot([0, 1], [0, 1], color=GREY, lw=0.5, ls='--'); ax.legend(fontsize=6.5, frameon=False, loc='lower right', handlelength=1.2, borderaxespad=0.2); ax.set_aspect('equal')
sty(ax, 'false positive rate', 'true positive rate'); fin(fig, 'f113.png')
# 11.6-1 pipeline schematic
fig, ax = plt.subplots(figsize=(2.05, 1.5), dpi=300); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 7)
def box(x, y, w, h, t, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08', fc=fc, ec=NAVY, lw=0.6)); ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=6.6)
box(0.2, 4.2, 2.6, 1.6, 'training\ndata', '#FFFFFF'); box(3.6, 4.2, 2.6, 1.6, 'scaler\nfit, transform', '#DCE8F8'); box(7.0, 4.2, 2.8, 1.6, 'model\nfit', '#FDEBD3')
box(0.2, 0.8, 2.6, 1.6, 'new\ndata', '#FFFFFF'); box(3.6, 0.8, 2.6, 1.6, 'scaler\ntransform', '#DCE8F8'); box(7.0, 0.8, 2.8, 1.6, 'model\npredict', '#FDEBD3')
for yy in (5.0, 1.6):
    for x0, x1 in ((2.85, 3.55), (6.25, 6.95)): ax.annotate('', (x1, yy), (x0, yy), arrowprops=dict(arrowstyle='->', lw=0.6, color=NAVY))
ax.annotate('', (4.9, 2.45), (4.9, 4.15), arrowprops=dict(arrowstyle='->', lw=0.5, color=GREY, ls='--')); ax.annotate('', (8.4, 2.45), (8.4, 4.15), arrowprops=dict(arrowstyle='->', lw=0.5, color=GREY, ls='--'))
ax.text(5.1, 3.2, 'means, scales', fontsize=6.5, color=GREY); ax.text(8.6, 3.2, 'weights', fontsize=6.5, color=GREY); fin(fig, 'f114.png')
