# Numbers quoted in the model answers to the questions for discussion (Chapter 53 solutions)
import sys, time, warnings
sys.path.insert(0, '/home/claude/mlbook3/ch53/code')
import numpy as np
from scipy import stats
from bcdata import load
from sklearn.feature_selection import (RFE, SelectKBest, SelectFdr, SelectFromModel, SelectFpr,
                                       SequentialFeatureSelector, f_classif)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

Xa, Xb, ya, yb, names = load(); Z = StandardScaler().fit_transform(Xa)
lr = lambda: LogisticRegression(max_iter=1000)

print("== Q1: the five columns of the filter and of RFE")
F, _ = f_classif(Z, ya); top = np.argsort(-F)[:5]
rfe = np.flatnonzero(RFE(lr(), n_features_to_select=5).fit(Z, ya).get_support())
R = np.abs(np.corrcoef(Z.T))
for nm, S in (("filter", np.sort(top)), ("RFE", rfe)):
    pr = [R[i, j] for a, i in enumerate(S) for j in S[a + 1:]]
    cv = cross_val_score(make_pipeline(StandardScaler(), lr()), Xa[:, S], ya, cv=5).mean()
    print(f"{nm:7s} columns {S.tolist()}  mean |r| between them {np.mean(pr):.3f}  max {np.max(pr):.3f}  CV on these columns {cv:.4f}")
print("filter names:", [str(names[i]) for i in np.sort(top)]); print("RFE names   :", [str(names[i]) for i in rfe])
print("ranks by F of the RFE columns:", [int(np.flatnonzero(np.argsort(-F) == j)[0]) + 1 for j in rfe])
print("pairs |r|>0.9 among the 5 best by F:", sum(R[i, j] > 0.9 for a, i in enumerate(top) for j in top[a + 1:]), "of 10")
Fv = np.sort(F)[::-1]; print("F of the best 5:", Fv[:5].round(1).tolist())
Xp = StandardScaler().fit_transform(Xa)[:, top]; w = lr().fit(Xp, ya).coef_[0]; print("weights of the filter's five in a joint fit:", w.round(2).tolist())

print("\n== Q2: SelectFdr, 40 columns kept; wrong ones as a function of m (10 per cent of columns shifted)")
def fdr_run(m, shift=0.9, n=100, reps=100, alpha=0.05):
    rng = np.random.RandomState(0); m1 = m // 10; kept = wrong = 0; frac = []
    for _ in range(reps):
        y = np.repeat([0, 1], n // 2); X = rng.randn(n, m); X[y == 1, :m1] += shift
        s = SelectFdr(f_classif, alpha=alpha).fit(X, y).get_support(); k = s.sum(); w = s[m1:].sum()
        kept += k; wrong += w; frac.append(w / max(k, 1))
    return kept / reps, wrong / reps, np.mean(frac)
for m in (200, 2000):
    k, w, f = fdr_run(m); print(f"m={m:5d}: kept {k:6.1f}  wrong {w:5.2f}  mean share wrong {f:.3f}  (alpha*m0/m = {0.05 * 0.9:.3f})")
print("bound on the expected number wrong among 40 kept: 0.05*40 =", 0.05 * 40, " with 10 per cent shifted: 0.045*40 =", 0.045 * 40)

print("\n== Q4: 50 000 columns, 200 rows")
p, n, s, ncv = 50000, 200, 10, 5
print("filter cost n*p =", n * p)
print("RFE step=1 to 10 columns, fits:", p - s + 1)
print("RFE step=0.1 (int(0.1*p) = 5000 per round), fits:", int(np.ceil((p - s) / int(0.1 * p))) + 1)
print("forward selection to 10 columns, fits:", ncv * (s * p - s * (s - 1) // 2))
rng = np.random.RandomState(0); X = rng.randn(n, p); y = np.repeat([0, 1], n // 2)
t = time.time(); Fs, ps_ = f_classif(X, y); dt = time.time() - t
print(f"f_classif on 200 x 50000 noise: {dt:.2f} s; best F {Fs.max():.1f}; smallest p {ps_.min():.1e}; columns with p<0.05: {(ps_ < .05).sum()} (expected {0.05 * p:.0f})")
print("largest |r| expected for pure noise: sqrt(2 ln m / n) =", round(np.sqrt(2 * np.log(p) / n), 3))
r = np.abs([np.corrcoef(X[:, j], y)[0, 1] for j in range(2000)]); print("largest |r| in the first 2000 columns:", r.max().round(3), " sqrt(2 ln 2000/200) =", round(np.sqrt(2 * np.log(2000) / n), 3))

class Counter(LogisticRegression):          # counts the fits made by RFE
    n_fit = 0
    def fit(self, X, y, **kw):
        Counter.n_fit += 1; return super().fit(X, y, **kw)
Xs = rng.randn(60, 1000); ys = np.repeat([0, 1], 30)
for st in (1, 0.1):
    Counter.n_fit = 0; RFE(Counter(max_iter=200), n_features_to_select=10, step=st).fit(Xs, ys)
    print(f"RFE p=1000 to 10, step={st}: fits counted {Counter.n_fit}; formula ceil(990/step_int)+1 =", int(np.ceil(990 / (1 if st == 1 else int(st * 1000)))) + 1)
