# Numbers quoted in the text of Chapter 24 (not shown as a listing).
import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import DecisionTreeRegressor as DTR


def gini(c):
    p = np.asarray(c, float) / np.sum(c)
    return 1 - np.sum(p ** 2)


def ent(c):
    p = np.asarray(c, float) / np.sum(c)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def mis(c):
    return 1 - np.max(c) / np.sum(c)


print("== impurity measures, parent (4,4), splits A and B")
for name, f in (("misclass", mis), ("gini", gini), ("entropy", ent)):
    parent = f([4, 4])
    A = (4 * f([3, 1]) + 4 * f([1, 3])) / 8
    B = (6 * f([4, 2]) + 2 * f([0, 2])) / 8
    print(f"{name:9s} parent {parent:.4f}  A {A:.4f}  B {B:.4f}"
          f"  decrease A {parent - A:.4f}  B {parent - B:.4f}")
print("max gini, entropy, misclass at p=1/2:",
      gini([1, 1]), ent([1, 1]), mis([1, 1]))
print("K=3 equal: gini", round(gini([1, 1, 1]), 4),
      "entropy", round(ent([1, 1, 1]), 4))

print("== breast cancer, gini versus entropy, full trees")
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
for crit in ("gini", "entropy"):
    m = DTC(criterion=crit, random_state=0).fit(Xtr, ytr)
    print(crit, "depth", m.get_depth(), "leaves", m.get_n_leaves(),
          "test", round(m.score(Xte, yte), 4))
accs = [DTC(random_state=s).fit(Xtr, ytr).score(Xte, yte)
        for s in range(10)]
print("test accuracy of full tree over 10 seeds: min",
      round(min(accs), 4), "max", round(max(accs), 4))

print("== class_weight on an imbalanced sample")
Xi, yi = make_classification(n_samples=600, n_features=4, n_informative=3,
                             n_redundant=0, weights=[0.9, 0.1],
                             random_state=0)
for cw in (None, "balanced"):
    m = DTC(max_depth=2, class_weight=cw, random_state=0).fit(Xi, yi)
    print(cw, "root value", m.tree_.value[0, 0].round(3),
          " leaves", m.get_n_leaves())

print("== monotonic constraint")
rng = np.random.default_rng(0)
x = np.sort(rng.uniform(0, 10, 80))
yy = 0.3 * x + rng.normal(0, 1.5, 80)
grid = np.linspace(0, 10, 400)[:, None]
for cst in (None, [1]):
    m = DTR(max_depth=5, monotonic_cst=cst).fit(x[:, None], yy)
    ok = bool(np.all(np.diff(m.predict(grid)) >= 0))
    print("monotonic_cst", cst, " non-decreasing:", ok)

print("== random splitter")
for sp in ("best", "random"):
    m = DTC(splitter=sp, random_state=0).fit(Xtr, ytr)
    print(sp, "depth", m.get_depth(), "leaves", m.get_n_leaves(),
          "test", round(m.score(Xte, yte), 4))

print("== the right node of the loan tree: best income split")
age = np.array([35, 27, 22, 48, 33, 24, 52, 29, 41, 58])
inc = np.array([22, 40, 55, 28, 62, 18, 36, 26, 48, 75])
yl = np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 1])
m = inc > 32
vals = np.unique(inc[m])
for t in (vals[:-1] + vals[1:]) / 2:
    L, R = yl[m][inc[m] <= t], yl[m][inc[m] > t]
    g = (len(L) * gini(np.bincount(L, minlength=2))
         + len(R) * gini(np.bincount(R, minlength=2))) / m.sum()
    print(f"  income <= {t:5.1f}: weighted gini {g:.4f}")

print("== cost-complexity of the three loan trees at alpha = 0.2")
for leaves, R in ((3, 0.0), (2, 1 / 6), (1, 0.5)):
    print(leaves, "leaves  R(T) =", round(R, 4), " R_alpha =",
          round(R + 0.2 * leaves, 4))

print("== scaling invariance on breast cancer")
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(Xtr)
a = DTC(random_state=0).fit(Xtr, ytr)
b = DTC(random_state=0).fit(sc.transform(Xtr), ytr)
print("same test predictions after standardising:",
      bool(np.array_equal(a.predict(Xte), b.predict(sc.transform(Xte)))),
      " same node count:", a.tree_.node_count == b.tree_.node_count)
print("depth/leaves raw", a.get_depth(), a.get_n_leaves(),
      " scaled", b.get_depth(), b.get_n_leaves())

print("== depth of unlimited trees on the moons of Listing 24.7")
from sklearn.datasets import make_moons
from sklearn.model_selection import StratifiedKFold
Xm, ym = make_moons(n_samples=300, noise=0.3, random_state=0)
Xmt, Xme, ymt, yme = train_test_split(
    Xm, ym, test_size=0.33, random_state=0, stratify=ym)
full = DTC(random_state=0).fit(Xmt, ymt)
print("training set: depth", full.get_depth(), "leaves", full.get_n_leaves())
print("folds:", [DTC(random_state=0).fit(Xmt[a], ymt[a]).get_depth()
                 for a, b in StratifiedKFold(5).split(Xmt, ymt)])

print("== exclusive-or of Figure 24.11-1: gains inside the two halves")
rng = np.random.default_rng(3)
Xx = rng.uniform(-1, 1, (240, 2))
yx = ((Xx[:, 0] > 0) != (Xx[:, 1] > 0)).astype(int)
yx = np.where(rng.random(240) < 0.06, 1 - yx, yx)


def g2(v):
    return 0.0 if len(v) == 0 else 2 * v.mean() * (1 - v.mean())


def gain(idx, j, t):
    L, R = idx[Xx[idx, j] <= t], idx[Xx[idx, j] > t]
    return len(idx) / 240 * (g2(yx[idx]) - len(L) / len(idx) * g2(yx[L])
                             - len(R) / len(idx) * g2(yx[R]))


allx = np.arange(240)
ts = np.linspace(-0.9, 0.9, 181)
left, right = allx[Xx[:, 0] <= 0], allx[Xx[:, 0] > 0]
print("root impurity", round(g2(yx), 4), " best first question x1, x2:",
      round(max(gain(allx, 0, t) for t in ts), 4),
      round(max(gain(allx, 1, t) for t in ts), 4))
print("best second question in the left half:",
      round(max(gain(left, 1, t) for t in ts), 4),
      " in the right half:", round(max(gain(right, 1, t) for t in ts), 4))
