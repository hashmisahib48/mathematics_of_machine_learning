# Question 3 of Chapter 56: how well do scores on few rows predict scores on all rows? (72-cell SVC grid)
import sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../ch56/code')
from common import cancer_split, svc_pipe, GRID
from scipy.stats import spearmanr
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, cross_val_score, train_test_split

Xa, Xt, ya, yt = cancer_split()
cands = list(ParameterGrid(GRID))
def score_all(rows, seed):
    """five-fold accuracy of every candidate on a stratified subsample of `rows` rows"""
    if rows < len(ya): idx, _ = train_test_split(np.arange(len(ya)), train_size=rows, random_state=seed, stratify=ya)
    else: idx = np.arange(len(ya))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return np.array([cross_val_score(svc_pipe().set_params(**p), Xa[idx], ya[idx], cv=cv).mean() for p in cands])
full = score_all(len(ya), 0)
order = np.argsort(-full, kind='stable'); top20 = order[:20]; b = int(order[0])
print("full data (426 rows): best five-fold accuracy %.4f; %d cells score at least 0.98" % (full.max(), (full >= 0.98).sum()))
print("\n rows  Spearman, all 72  Spearman, best 20  distinct scores  best cell kept by the best 1/3 | best 1/9  (mean of 5 subsamples)")
for rows in (26, 53, 106, 213):
    rho, rho20, dist, k3, k9 = [], [], [], [], []
    for seed in range(5):
        s = score_all(rows, seed); rank = np.argsort(np.argsort(-s, kind='stable'), kind='stable')
        rho.append(spearmanr(s, full)[0]); rho20.append(spearmanr(s[top20], full[top20])[0]); dist.append(len(np.unique(s.round(6))))
        k3.append(rank[b] < 24); k9.append(rank[b] < 8)
    print("%4d   %.2f              %.2f             %4.1f             %.1f | %.1f" % (rows, np.mean(rho), np.mean(rho20), np.mean(dist), np.mean(k3), np.mean(k9)))
