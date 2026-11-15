# Questions 2 and 4 of Chapter 56: the tied cells of the 72-cell grid, defaults, nested score, test score
import sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../ch56/code')
from common import cancer_split, svc_pipe, GRID
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

Xa, Xt, ya, yt = cancer_split()
gs = GridSearchCV(svc_pipe(), GRID, cv=5).fit(Xa, ya)
r = gs.cv_results_; mean = r['mean_test_score']; std = r['std_test_score']
C = np.log2([p['svc__C'] for p in r['params']]).astype(int); G = np.log2([p['svc__gamma'] for p in r['params']]).astype(int)
best = mean.max(); tied = np.flatnonzero(mean == best)
print("Q2  best five-fold accuracy %.4f, reached by %d cells; best_index_ = %d" % (best, len(tied), gs.best_index_))
sc = {}
for i in tied:
    nb = [j for j in range(72) if abs(C[j] - C[i]) <= 2 and abs(G[j] - G[i]) <= 2]
    cell = svc_pipe().set_params(svc__C=2.0 ** C[i], svc__gamma=2.0 ** G[i]).fit(Xa, ya)
    print("  cell %2d  log2 C = %2d  log2 gamma = %3d  mean %.4f  fold sd %.4f  neighbours (%d cells) mean %.4f min %.4f  test %.4f"
          % (i, C[i], G[i], mean[i], std[i], len(nb), mean[nb].mean(), mean[nb].min(), cell.score(Xt, yt)))
print("  one patient of %d = %.4f" % (len(ya), 1 / len(ya)))
print("  rank_test_score of the two cells:", r['rank_test_score'][tied].tolist())

print("\nQ4")
d = cross_val_score(svc_pipe(), Xa, ya, cv=StratifiedKFold(5)).mean()
print("  defaults: five-fold %.4f (cross_val_score, stratified)   test %.4f" % (d, svc_pipe().fit(Xa, ya).score(Xt, yt)))
print("  grid: best_score_ %.4f (of M = 72)   test %.4f   (%d of %d test patients)" % (gs.best_score_, gs.score(Xt, yt), round(gs.score(Xt, yt) * len(yt)), len(yt)))
print("  difference of the five-fold scores %.4f = %.1f patients of %d" % (gs.best_score_ - d, (gs.best_score_ - d) * len(ya), len(ya)))
nest = cross_val_score(GridSearchCV(svc_pipe(), GRID, cv=5), Xa, ya, cv=StratifiedKFold(5, shuffle=True, random_state=0))
print("  nested, five outer folds:", nest.round(4).tolist(), " mean %.4f" % nest.mean())
p = gs.score(Xt, yt); print("  standard error of the test accuracy on %d patients: %.4f" % (len(yt), np.sqrt(p * (1 - p) / len(yt))))
