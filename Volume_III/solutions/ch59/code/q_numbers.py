# Numbers quoted in the model answers to the four questions for discussion
import sys
import numpy as np
sys.path.insert(0, "../../../ch59/code")
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import silhouette_score as sil
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (cross_val_score, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

print("== Q1  baselines and the dependence on the question (breast cancer, 5-fold on the 426 training rows)")
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
for name, m in (("constant", DummyClassifier()),
                ("scaled logistic", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)))):
    print("%-16s %.4f" % (name, cross_val_score(m, Xtr, ytr, cv=5).mean()))
Xw, yw = load_wine(return_X_y=True)
print("wine, k-means K=3: raw ARI %.3f, standardised ARI %.3f" % (
    ari(yw, KMeans(3, n_init=10, random_state=0).fit_predict(Xw)),
    ari(yw, KMeans(3, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(Xw)))))

print("\n== Q2  peeking at the test rows: 50 equally good candidates, n = 143 test rows")
rng = np.random.RandomState(0)
n, true_acc, n_cand, reps = 143, 0.95, 50, 20000
tests = rng.binomial(n, true_acc, size=(reps, n_cand)) / n
best = tests.max(1)
print("true accuracy of every candidate %.3f; mean test accuracy of the best of %d: %.4f (sd %.4f)"
      % (true_acc, n_cand, best.mean(), best.std()))
print("a fresh test set of 143 rows for the chosen one: mean %.4f" % (rng.binomial(n, true_acc, reps) / n).mean())
print("optimism %.4f = %.1f patients" % (best.mean() - true_acc, (best.mean() - true_acc) * n))
print("same with 5 candidates: %.4f" % (rng.binomial(n, true_acc, size=(reps, 5)) / n).max(1).mean())

print("\n== Q3  log price or price: ridge and boosting, five-fold out-of-fold predictions on the training part")
from case_b1 import Xtr as hX, ytr as hy, ridge, hgb, log
for name, m in (("ridge", ridge()), ("ridge, log", log(ridge())), ("boosting", hgb()), ("boosting, log", log(hgb()))):
    p = cross_val_predict(m, hX, hy, cv=5)
    rel = np.median(np.abs(p - hy) / hy)
    print("%-14s RMSE %6.1f  MAE %5.1f  median rel. err %.3f  mean pred %.1f (mean price %.1f)  bias %.1f%%"
          % (name, np.sqrt(mean_squared_error(hy, p)), np.mean(np.abs(p - hy)), rel, p.mean(), hy.mean(),
             100 * (p.mean() / hy.mean() - 1)))
# smearing: scale the back-transformed prediction by the mean ratio of actual to predicted (training residuals)
m = log(ridge()); p = cross_val_predict(m, hX, hy, cv=5)
s = np.mean(hy / p)
print("ridge, log: smearing factor %.3f -> RMSE %.1f, bias %.1f%%" % (s, np.sqrt(mean_squared_error(hy, s * p)),
      100 * ((s * p).mean() / hy.mean() - 1)))
big = hy > hy.quantile(0.9)
print("ridge, log: share of the squared error from the dearest 10%% of houses: %.2f; raw ridge: %.2f" % (
    ((p - hy)[big] ** 2).sum() / ((p - hy) ** 2).sum(),
    (lambda q: ((q - hy)[big] ** 2).sum() / ((q - hy) ** 2).sum())(cross_val_predict(ridge(), hX, hy, cv=5))))

print("\n== Q4  wine: silhouette against ARI, and a stability check")
Z = PCA(2, random_state=0).fit_transform(StandardScaler().fit_transform(Xw))
km = KMeans(3, n_init=10, random_state=0).fit_predict(Z)
hd = HDBSCAN(min_cluster_size=10, copy=True).fit_predict(Z)
keep = hd >= 0
print("k-means: silhouette %.3f ARI %.3f" % (sil(Z, km), ari(yw, km)))
print("HDBSCAN: noise %d of %d, silhouette on the kept %.3f, ARI %.3f" % ((~keep).sum(), len(Z), sil(Z[keep], hd[keep]), ari(yw, hd)))
print("k-means silhouette on the same kept wines %.3f" % sil(Z[keep], km[keep]))
print("ARI of the HDBSCAN clusters with the cultivars on the kept wines only: %.3f" % ari(yw[keep], hd[keep]))
print("cluster sizes of HDBSCAN:", np.bincount(hd[keep]).tolist(), " cultivar sizes:", np.bincount(yw).tolist())
rs = np.random.RandomState(1); aris_k, aris_h = [], []
for _ in range(20):
    i = rs.choice(len(Z), len(Z), replace=True); j = rs.choice(len(Z), len(Z), replace=True)
    a = KMeans(3, n_init=5, random_state=0).fit(Z[i]).predict(Z); b = KMeans(3, n_init=5, random_state=0).fit(Z[j]).predict(Z)
    aris_k.append(ari(a, b))
print("k-means stability: mean ARI between fits on two bootstrap resamples %.3f (min %.3f)" % (np.mean(aris_k), np.min(aris_k)))
