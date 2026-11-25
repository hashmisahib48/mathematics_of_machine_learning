"""Sidebar data figures for the five problems of Chapter 11 (no answers are drawn)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from figstyle import *
import statsmodels.api as sm
from sklearn.datasets import load_wine, load_diabetes, load_breast_cancer
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_val_score, GridSearchCV, train_test_split, StratifiedKFold)
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
P = lambda n: os.path.join(HERE, 'png', n)
FS, TS = 7.2, 6.8
def fin(fig, n): fig.tight_layout(pad=0.25); fig.savefig(P(n), dpi=300, facecolor='white'); plt.close(fig)
def sty(ax, xl, yl): ax.set_xlabel(xl, fontsize=FS); ax.set_ylabel(yl, fontsize=FS); ax.tick_params(labelsize=TS, length=2.2)
# P11-1 wines: the 50 fold accuracies of each classifier
X, y = load_wine(return_X_y=True); cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
M = [('k-NN', make_pipeline(StandardScaler(), KNeighborsClassifier(5))), ('logistic', make_pipeline(StandardScaler(), LogisticRegression())),
     ('naive Bayes', GaussianNB()), ('tree', DecisionTreeClassifier(max_depth=3, random_state=0))]
S = [cross_val_score(m, X, y, cv=cv) for _, m in M]
fig, ax = plt.subplots(figsize=(2.05, 1.7), dpi=300)
bp = ax.boxplot(S[::-1], orientation='horizontal', widths=0.55, patch_artist=True, showfliers=True, flierprops=dict(ms=2.2, mfc=GREY, mec='none'),
                medianprops=dict(color=NAVY, lw=1.0), boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
for b, c in zip(bp['boxes'], [OR, GREEN, RED, BLUE]): b.set_facecolor(c); b.set_alpha(0.55)
ax.set_yticklabels([n for n, _ in M][::-1]); sty(ax, 'fold accuracy', ''); fin(fig, 'pj111.png')
# P11-2 diabetes: cross-validated R2 over degree and penalty
X, y = load_diabetes(return_X_y=True); a, b, c, d = train_test_split(X, y, test_size=0.25, random_state=0)
p = Pipeline([('poly', PolynomialFeatures(include_bias=False)), ('sc', StandardScaler()), ('r', Ridge())])
al = np.logspace(-2, 3, 11)
g = GridSearchCV(p, {'poly__degree': [1, 2, 3], 'r__alpha': al}, cv=5).fit(a, c)
Z = g.cv_results_['mean_test_score'].reshape(3, 11); print('11-2 grid R2 min max', Z.min().round(3), Z.max().round(3))
fig, ax = plt.subplots(figsize=(2.05, 1.5), dpi=300)
im = ax.imshow(np.clip(Z, -0.2, None), cmap='viridis', aspect='auto', origin='lower', extent=(-2.25, 3.25, 0.5, 3.5), vmin=-0.2)
ax.set_yticks([1, 2, 3]); ax.set_xticks([-2, 0, 2]); sty(ax, r'$\log_{10}$ penalty', 'degree')
cb = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.06); cb.ax.tick_params(labelsize=TS, length=2); cb.set_label(r'CV $R^2$ (clipped at $-0.2$)', fontsize=6.5)
fin(fig, 'pj112.png')
# P11-3 breast cancer: inner cross-validation curves of the k-NN grid
X, y = load_breast_cancer(return_X_y=True)
pipe = Pipeline([('s', StandardScaler()), ('k', KNeighborsClassifier())]); ks = [1, 3, 5, 7, 9, 15, 25]
gs = GridSearchCV(pipe, {'k__n_neighbors': ks, 'k__weights': ['uniform', 'distance']}, cv=StratifiedKFold(5, shuffle=True, random_state=1)).fit(X, y)
r = gs.cv_results_; fig, ax = plt.subplots(figsize=(2.05, 1.6), dpi=300)
for w, col in (('uniform', BLUE), ('distance', RED)):
    m = [i for i in range(len(r['params'])) if r['params'][i]['k__weights'] == w]
    ax.plot([r['params'][i]['k__n_neighbors'] for i in m], [r['mean_test_score'][i] for i in m], 'o-', ms=2.6, lw=0.9, color=col, label=w)
ax.set_ylim(0.948, 0.9765); ax.legend(fontsize=6.5, frameon=False, loc='upper right', ncol=2, columnspacing=0.8); sty(ax, 'number of neighbors $k$', 'inner CV accuracy'); fin(fig, 'pj113.png')
# P11-4 grades: the 32 students
D = sm.datasets.spector.load_pandas().data; D.to_csv(os.path.join(HERE, '..', 'data', 'spector.csv'), index=False)
fig, ax = plt.subplots(figsize=(2.05, 1.7), dpi=300)
for gr, psi, mk, col, lab in ((0, 0, 'o', BLUE, 'no/old'), (0, 1, '^', BLUE, 'no/new'), (1, 0, 'o', OR, 'yes/old'), (1, 1, '^', OR, 'yes/new')):
    q = D[(D.GRADE == gr) & (D.PSI == psi)]; ax.plot(q.GPA, q.TUCE, mk, ms=3.6, color=col, mec='white', mew=0.3, label=lab)
ax.legend(fontsize=6.5, frameon=False, loc='upper left', ncol=2, columnspacing=0.8, handletextpad=0.1, borderaxespad=0.1); ax.set_ylim(10, 40)
sty(ax, 'grade point average', 'TUCE score'); fin(fig, 'pj114.png')
print('ok')
