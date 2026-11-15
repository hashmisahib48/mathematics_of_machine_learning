# Hand checks for Problems 29A.1 - 29A.4 (every number quoted in the solutions)
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from math import comb
from sklearn.datasets import load_wine
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

print("== 29A.1  five independent voters, p = 0.8")
p = 0.8
terms = {j: comb(5, j) * p**j * (1 - p)**(5 - j) for j in range(6)}
for j, t in terms.items(): print(f"j = {j}: C(5,{j}) = {comb(5, j)}  term {t:.5f}")
print("P(majority right), M = 5:", sum(terms[j] for j in (3, 4, 5)))
print("M = 3, Eq. 29.3-4:", 3 * p**2 - 2 * p**3, "  by terms:", sum(comb(3, j) * p**j * (1 - p)**(3 - j) for j in (2, 3)))
print("gain 1 -> 3:", 3 * p**2 - 2 * p**3 - p, " gain 3 -> 5:", sum(terms[j] for j in (3, 4, 5)) - (3 * p**2 - 2 * p**3))
print("error of one voter 0.2, of three:", 1 - (3 * p**2 - 2 * p**3), " of five:", 1 - sum(terms[j] for j in (3, 4, 5)))
print("slip: counting j >= 2 for M = 5:", sum(terms[j] for j in (2, 3, 4, 5)))

print("\n== 29A.2  members 0.60, 0.55, 0.05; true class 1")
P = np.array([0.60, 0.55, 0.05]); w = np.array([1, 1, 3.])
print("votes above 1/2:", int((P > 0.5).sum()), " hard class:", int((P > 0.5).sum() > 1.5))
print("soft:", P.mean(), " class", int(P.mean() > 0.5))
print("weighted: numerator", P @ w, " / ", w.sum(), "=", P @ w / w.sum(), " class", int(P @ w / w.sum() > 0.5))
ll = -np.log(P); print("member log losses:", ll.round(4), " mean:", ll.mean().round(4))
print("soft vote log loss:", -np.log(P.mean()).round(4), " weighted vote log loss:", -np.log(P @ w / w.sum()).round(4))
print("log loss of the weighted vote vs weighted mean of members' losses:", (ll @ w / w.sum()).round(4))

print("\n== 29A.3  members 4, 7, 10; target 6")
f = np.array([4., 7., 10.]); y = 6.; fb = f.mean()
print("average", fb, " error of average", (fb - y)**2)
print("member errors", (f - y)**2, " mean", ((f - y)**2).mean())
print("deviations from the average", f - fb, " squares", (f - fb)**2, " spread", ((f - fb)**2).mean())
print("check 7 - 6 =", ((f - y)**2).mean() - ((f - fb)**2).mean())
print("cross term sum of (f - fb) (fb - y):", ((f - fb) * (fb - y)).sum())

print("\n== 29A.4  stacking on the wine data, cv = 7")
X, yw = load_wine(return_X_y=True)
class Count(LogisticRegression):
    n = 0
    def fit(self, X, y, **kw): Count.n += 1; return super().fit(X, y, **kw)
est = [("lr", make_pipeline(StandardScaler(), LogisticRegression())),
       ("svc", make_pipeline(StandardScaler(), SVC())),
       ("tree", DecisionTreeClassifier(random_state=0))]
st = StackingClassifier(est, cv=7).fit(X, yw)
print("stack_method_:", st.stack_method_)
print("width of Z:", st.transform(X).shape[1], " classes", len(st.classes_), " features", X.shape[1])
stp = StackingClassifier(est, cv=7, passthrough=True, final_estimator=make_pipeline(StandardScaler(), LogisticRegression())).fit(X, yw)
print("width with passthrough:", stp.transform(X).shape[1])
Count.n = 0
StackingClassifier([("a", Count()), ("b", Count()), ("c", Count())], cv=7).fit(StandardScaler().fit_transform(X), yw)
print("member fits counted (final estimator is the default, not counted):", Count.n)
M, K = 3, 7
print("M (K + 1) =", M * (K + 1), " M (K + 1) + 1 =", M * (K + 1) + 1, "; ordinary cost of voting:", M)
print("SVC has predict_proba:", hasattr(est[1][1], "predict_proba"), " (probability=False)")
