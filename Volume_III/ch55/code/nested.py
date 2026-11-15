# Winner's curse: pick the best of 20 useless candidates (NumPy only).
import json, numpy as np
from sklearn.model_selection import KFold

def fold_acc(X, y, tr, te):
    """Accuracy on te of 20 one-feature rules 'nearer class mean'."""
    m0, m1 = X[tr][y[tr] == 0].mean(0), X[tr][y[tr] == 1].mean(0)
    pred = np.abs(X[te] - m1) < np.abs(X[te] - m0)      # (n_te, 20)
    return (pred == y[te][:, None].astype(bool)).mean(0)

def cv_acc(X, y, rows, K, seed):
    """K-fold accuracy of each candidate, on the given rows."""
    sc = [fold_acc(X, y, rows[tr], rows[te]) for tr, te in
          KFold(K, shuffle=True, random_state=seed).split(rows)]
    return np.mean(sc, axis=0)

rng = np.random.RandomState(0)
best, nested, fresh = [], [], []
for rep in range(500):
    X, y = rng.normal(size=(60, 20)), rng.randint(0, 2, 60)
    allrows = np.arange(60)
    s = cv_acc(X, y, allrows, 5, rep)
    best.append(s.max())                         # optimistic winner
    outer = []
    for tr, te in KFold(5, shuffle=True, random_state=rep).split(X):
        j = cv_acc(X, y, tr, 4, rep).argmax()    # choose on train only
        outer.append(fold_acc(X, y, tr, te)[j])
    nested.append(np.mean(outer))
    # fresh data: the chosen rule on 5000 new samples
    j = s.argmax(); Xn, yn = rng.normal(size=(5000, 20)), rng.randint(0, 2, 5000)
    fresh.append(fold_acc(np.r_[X, Xn], np.r_[y, yn], allrows,
                          np.arange(60, 5060))[j])
q = lambda v: (round(float(np.mean(v)), 4), round(float(np.std(v, ddof=1)), 4))
out = {"best_cv": q(best), "nested": q(nested), "fresh": q(fresh),
       "best_cv_values": [round(float(v), 4) for v in best[:300]],
       "nested_values": [round(float(v), 4) for v in nested[:300]],
       "fresh_values": [round(float(v), 4) for v in fresh[:300]]}
print("labels are coin flips: true accuracy of any rule is 0.5")
for k in ("best_cv", "nested", "fresh"):
    print(f"{k:8s} mean {out[k][0]:.4f}  sd {out[k][1]:.4f}")
json.dump(out, open("verify/nested.json", "w"))
