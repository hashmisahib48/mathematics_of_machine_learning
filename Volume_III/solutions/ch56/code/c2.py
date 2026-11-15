# Problem 56C.2: uniform against log-uniform draws of C
import sys, os, math, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../ch56/code')
from scipy.stats import loguniform, uniform
from sklearn.model_selection import GridSearchCV

lo, hi = 1e-4, 100.0
print("share of the draws in each decade, C in (1e-4, 100):")
print("  decade      uniform (exact)   uniform (20 000 draws)   loguniform (exact)   loguniform (20 000 draws)")
rs = np.random.RandomState(0); u = uniform(lo, hi - lo).rvs(20000, random_state=rs); l = loguniform(lo, hi).rvs(20000, random_state=rs)
for a in range(-4, 2):
    ex_u = (10.0 ** (a + 1) - 10.0 ** a) / (hi - lo); ex_l = 1 / 6
    print("  1e%+d..1e%+d   %.4f            %.4f                  %.4f               %.4f" % (a, a + 1, ex_u, ((u >= 10.0 ** a) & (u < 10.0 ** (a + 1))).mean(), ex_l, ((l >= 10.0 ** a) & (l < 10.0 ** (a + 1))).mean()))
print("  uniform: P(C < 1) = %.4f   P(C < 0.01) = %.2e;   loguniform: P(C < 1) = %.3f" % ((1 - lo) / (hi - lo), (0.01 - lo) / (hi - lo), math.log(1 / lo) / math.log(hi / lo)))

# where are the good values of C for the logistic regression of Problem 56B.1?
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
X, y = load_breast_cancer(return_X_y=True)
Xa, Xt, ya, yt = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
Cs = np.logspace(-4, 2, 13)
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
gs = GridSearchCV(pipe, {"logisticregression__C": Cs}, cv=5).fit(Xa, ya); m = gs.cv_results_['mean_test_score']
good = Cs[m >= 0.97]; cl, ch = good.min(), good.max()
print("\nlogistic regression: C values with five-fold accuracy >= 0.97: %s" % np.round(good, 4).tolist())
pu = (ch - cl) / (hi - lo); pl = math.log(ch / cl) / math.log(hi / lo)
print("  good range %.3f to %.3f: share of the draws uniform %.4f, loguniform %.3f" % (cl, ch, pu, pl))
for nm, p in (("uniform", pu), ("loguniform", pl)):
    print("  %-10s: draws for probability 0.95 of at least one good value: %d;  with 20 draws: %.3f" % (nm, math.ceil(math.log(0.05) / math.log(1 - p)), 1 - (1 - p) ** 20))
