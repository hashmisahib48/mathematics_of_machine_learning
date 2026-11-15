# Checks of the leaf-value rules of Chapter 27 against scikit-learn.
import numpy as np
from scipy.special import expit, softmax
from sklearn.datasets import load_diabetes, load_iris, load_breast_cancer
from sklearn.ensemble import (GradientBoostingRegressor as GR,
                              GradientBoostingClassifier as GC)

X, y = load_diabetes(return_X_y=True)


def leaf_check(g, X, y, F, stage, rule):
    """max |tree value - rule(residuals in the leaf)| over the leaves."""
    t = g.estimators_[stage, 0]
    leaf = t.apply(X)
    res = y - F
    out = 0.0
    for l in np.unique(leaf):
        out = max(out, abs(t.tree_.value[l, 0, 0] - rule(res[leaf == l])))
    return out


def stage_F(g, X, m):                      # F_m (m = 0 is init)
    if m == 0:
        return np.full(len(X), g.init_.constant_.item())
    return list(g.staged_predict(X))[m - 1]


def pct(e, q):                  # scikit-learn's weighted percentile
    return np.percentile(e, q, method="inverted_cdf")


def huber_rule(delta):
    def rule(e):
        med = pct(e, 50)
        return med + np.mean(np.sign(e - med)
                             * np.minimum(delta, np.abs(e - med)))
    return rule


kw = dict(n_estimators=3, max_depth=2, learning_rate=0.5,
          random_state=0)
med = lambda e: pct(e, 50)
med.__name__ = "median"
for loss, rule in (("squared_error", np.mean), ("absolute_error", med)):
    g = GR(loss=loss, **kw).fit(X, y)
    d = [leaf_check(g, X, y, stage_F(g, X, m), m, rule) for m in range(3)]
    print(f"{loss:15s} leaf = {rule.__name__:6s} of residuals, "
          f"max error {max(d):.1e}")
g = GR(loss="quantile", alpha=0.8, **kw).fit(X, y)
rule = lambda e: pct(e, 80)
d = [leaf_check(g, X, y, stage_F(g, X, m), m, rule) for m in range(3)]
print(f"quantile        leaf = 80% quantile, max error {max(d):.1e}")
g = GR(loss="huber", alpha=0.9, **kw).fit(X, y)
d = []
for m in range(3):
    F = stage_F(g, X, m)
    delta = pct(np.abs(y - F), 90)
    d.append(leaf_check(g, X, y, F, m, huber_rule(delta)))
print(f"huber           leaf = Friedman's rule, max error {max(d):.1e}")
print("init:", GR(loss="quantile", alpha=0.8).fit(X, y).init_.constant_[0][0]
      .round(4), np.percentile(y, 80).round(4))

# Binomial: Newton leaf, scores, train_score_ = 2 x log loss.
Xc, yc = load_breast_cancer(return_X_y=True)
g = GC(n_estimators=2, max_depth=2, learning_rate=0.5, random_state=0)
g.fit(Xc, yc)
pi = yc.mean()
F0 = np.full(len(yc), np.log(pi / (1 - pi)))
p = expit(F0)
t = g.estimators_[0, 0]
leaf = t.apply(Xc)
err = max(abs(t.tree_.value[l, 0, 0] - (yc - p)[leaf == l].sum()
              / (p * (1 - p))[leaf == l].sum()) for l in np.unique(leaf))
print("binomial Newton leaf, max error", f"{err:.1e}")
F1 = F0 + 0.5 * t.tree_.value[leaf, 0, 0]
ll = -np.mean(yc * np.log(expit(F1)) + (1 - yc) * np.log(1 - expit(F1)))
print("train_score_[0] =", g.train_score_[0].round(6),
      " 2 x mean log loss =", (2 * ll).round(6))

# Exponential loss: leaf = sum(w s) / sum(w), w = exp(-s F), s = +-1.
g = GC(loss="exponential", n_estimators=1, max_depth=2,
       learning_rate=0.5, random_state=0).fit(Xc, yc)
F0 = np.full(len(yc), 0.5 * np.log(pi / (1 - pi)))
s = 2 * yc - 1
w = np.exp(-s * F0)
t = g.estimators_[0, 0]
leaf = t.apply(Xc)
err = max(abs(t.tree_.value[l, 0, 0] - (w * s)[leaf == l].sum()
              / w[leaf == l].sum()) for l in np.unique(leaf))
print("exponential: F0 =", g._raw_predict_init(Xc[:1])[0, 0].round(4),
      0.5 * np.log(pi / (1 - pi)).round(4), " leaf error", f"{err:.1e}")
print("  p = expit(2F):", np.allclose(
    g.predict_proba(Xc)[:, 1], expit(2 * g.decision_function(Xc))))

# Multiclass: K trees per stage, factor (K - 1)/K.
Xi, yi = load_iris(return_X_y=True)
Xi, yi = Xi[::2], yi[::2]
g = GC(n_estimators=1, max_depth=2, learning_rate=0.5, random_state=0)
g.fit(Xi, yi)
K = 3
P = np.full((len(yi), K), 1 / K)
print("iris: estimators_ shape", g.estimators_.shape)
for k in range(K):
    r = (yi == k) - P[:, k]
    t = g.estimators_[0, k]
    leaf = t.apply(Xi)
    err = max(abs(t.tree_.value[l, 0, 0] - (K - 1) / K * r[leaf == l].sum()
                  / (P[:, k] * (1 - P[:, k]))[leaf == l].sum())
              for l in np.unique(leaf))
    print(f"  class {k}: leaf = (K-1)/K x sum r / sum p(1-p), error {err:.1e}")
Xw = np.arange(60).reshape(-1, 1) * 1.0
yw = np.r_[np.zeros(30), np.ones(20), 2 * np.ones(10)].astype(int)
g = GC(n_estimators=1).fit(Xw, yw)
pr = np.bincount(yw) / len(yw)
print("init raw scores", g._raw_predict_init(Xw[:1]).round(4)[0],
      "log prior minus its mean", (np.log(pr) - np.log(pr).mean()).round(4))
