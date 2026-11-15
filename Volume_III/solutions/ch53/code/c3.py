# Problem 53C.3 and Question 3: which procedures give a biased cross-validation score?
# Pure noise, n = 60 rows, m = 1000 columns, labels 0,1,0,1,...: every honest score is 0.5.
import numpy as np
from sklearn.feature_selection import RFECV, SelectKBest, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

n, m, reps = 60, 1000, 15
ks = [1, 2, 5, 10, 20, 50, 100, 200]
res = {key: [] for key in ("a", "b", "c", "c_own", "d_best", "d_mean", "d_nested")}
for seed in range(reps):
    rng = np.random.RandomState(seed)
    X, y = rng.standard_normal((n, m)), np.arange(n) % 2
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    lr = LogisticRegression()
    # (a) variance filter on all rows (no labels used): keep the half with larger variance
    thr = np.median(X.var(axis=0))
    res["a"].append(cross_val_score(lr, VarianceThreshold(thr).fit_transform(X), y, cv=cv).mean())
    # (b) F filter on all rows, then cross-validate the classifier
    res["b"].append(cross_val_score(lr, SelectKBest(k=10).fit_transform(X, y), y, cv=cv).mean())
    # (c) RFECV as the first step of a pipeline; the outer folds are untouched by it
    rfecv = RFECV(LogisticRegression(), step=0.1, cv=3)
    res["c"].append(cross_val_score(make_pipeline(rfecv, lr), X, y, cv=cv).mean())
    res["c_own"].append(rfecv.fit(X, y).cv_results_["mean_test_score"].max())   # its own best score
    # (d) k chosen by cross-validation, the best score reported
    sc = [cross_val_score(make_pipeline(SelectKBest(k=k), lr), X, y, cv=cv).mean() for k in ks]
    res["d_best"].append(max(sc)); res["d_mean"].append(np.mean(sc))
    grid = GridSearchCV(make_pipeline(SelectKBest(), lr), {"selectkbest__k": ks}, cv=3)
    res["d_nested"].append(cross_val_score(grid, X, y, cv=cv).mean())
lab = {"a": "(a) variance filter on all rows", "b": "(b) SelectKBest on all rows",
       "c": "(c) RFECV in a pipeline, outer CV", "c_own": "(c') RFECV's own best score",
       "d_best": "(d) best of 8 values of k", "d_mean": "    mean over the 8 values of k",
       "d_nested": "(d') k chosen in an inner CV"}
print("accuracy on noise, mean +- sd over", reps, "repetitions (honest value 0.500)")
for k, v in res.items(): print(f"{lab[k]:36s} {np.mean(v):.3f} +- {np.std(v):.3f}")
