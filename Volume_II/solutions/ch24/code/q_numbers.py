# Numbers quoted in the model answers to the questions for discussion of Chapter 24.
import sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../ch24/code')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier as DTC
gini = lambda c: 1 - ((np.array(c) / sum(c)) ** 2).sum()
ent = lambda c: -sum(q * np.log2(q) for q in np.array(c) / sum(c) if q > 0)
mis = lambda c: 1 - max(c) / sum(c)
def dec(f, parent, kids):
    n = sum(sum(k) for k in kids); return f(parent) - sum(sum(k) / n * f(k) for k in kids)

print("== Q1  a split on which the misclassification rate is blind")
for name, f in (("misclass", mis), ("gini", gini), ("entropy", ent)):
    print("%-9s parent (6,2) = %.4f   split (3,0)|(3,2): decrease %.4f    split (4,0)|(2,2): decrease %.4f"
          % (name, f([6, 2]), dec(f, [6, 2], [[3, 0], [3, 2]]), dec(f, [6, 2], [[4, 0], [2, 2]])))
print("Gini of (3,2) = %.4f; weighted children (5/8)(0.48) = %.4f" % (gini([3, 2]), 5 / 8 * gini([3, 2])))
print("loan root, income <= 32: misclass decrease %.4f gini %.4f" % (dec(mis, [5, 5], [[4, 0], [1, 5]]), dec(gini, [5, 5], [[4, 0], [1, 5]])))
print("loan root, age <= 25.5 : misclass decrease %.4f gini %.4f" % (dec(mis, [5, 5], [[2, 0], [3, 5]]), dec(gini, [5, 5], [[2, 0], [3, 5]])))

print("\n== Q2  scaling and logarithm of the income of the loan data")
age = np.array([35, 27, 22, 48, 33, 24, 52, 29, 41, 58.]); inc = np.array([22, 40, 55, 28, 62, 18, 36, 26, 48, 75.]); y = np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 1])
new = np.array([[30, 45], [23, 60.]])
for name, f in (("raw", lambda v: v), ("x1000", lambda v: v * 1000), ("log", np.log)):
    X = np.c_[age, f(inc)]; t = DTC(random_state=0).fit(X, y); Xn = np.c_[new[:, 0], f(new[:, 1])]
    thr = t.tree_.threshold[0]; back = thr if name == "raw" else thr / 1000 if name == "x1000" else np.exp(thr)
    print("%-6s tree: root feature %d threshold %.4g (income scale %.3f), leaves %d, predictions A, B: %s, same partition of training set: %s"
          % (name, t.tree_.feature[0], thr, back, t.get_n_leaves(), t.predict(Xn).tolist(), np.array_equal(t.apply(X), DTC(random_state=0).fit(np.c_[age, inc], y).apply(np.c_[age, inc]))))
for name, f in (("raw", lambda v: v), ("x1000", lambda v: v * 1000)):
    X = np.c_[age, f(inc)]; kn = KNeighborsClassifier(3).fit(X, y); Xn = np.c_[new[:, 0], f(new[:, 1])]
    d = np.linalg.norm(X - Xn[0], axis=1); print("3-NN %-6s: predictions A, B %s; neighbours of A: %s" % (name, kn.predict(Xn).tolist(), (np.argsort(d)[:3] + 1).tolist()))
print("midpoint of incomes 28 and 36: linear %.3f, geometric (back from logs) %.3f" % ((28 + 36) / 2, np.exp((np.log(28) + np.log(36)) / 2)))
print("midpoint of ages 22 and 24: %.1f" % 23)

print("\n== Q3  training accuracy along the pruning path (breast cancer)")
d = load_breast_cancer(); Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=0.25, random_state=0, stratify=d.target)
path = DTC(random_state=0).cost_complexity_pruning_path(Xtr, ytr); al = path.ccp_alphas[:-1]
tr, cv, te, lv = [], [], [], []
for a in al:
    m = DTC(ccp_alpha=a, random_state=0).fit(Xtr, ytr); tr.append(m.score(Xtr, ytr)); te.append(m.score(Xte, yte)); lv.append(m.get_n_leaves())
    cv.append(cross_val_score(DTC(ccp_alpha=a, random_state=0), Xtr, ytr, cv=5).mean())
tr, cv, te = map(np.array, (tr, cv, te))
print("number of alphas %d; training accuracy nonincreasing along the path: %s" % (len(al), bool(np.all(np.diff(tr) <= 1e-12))))
print("training accuracy at alpha = 0: %.4f (%d leaves); argmax of training accuracy is alpha = %.4f" % (tr[0], lv[0], al[int(np.argmax(tr))]))
b = int(np.argmax(cv)); print("validation maximum %.4f at alpha %.4f (%d leaves); training accuracy there %.4f; test %.4f" % (cv[b], al[b], lv[b], tr[b], te[b]))
print("validation at alpha = 0: %.4f" % cv[0])

print("\n== Q4  the exclusive-or: effective alpha of the root")
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]); yx = np.array([0, 1, 1, 0])
t = DTC(random_state=0); p = t.cost_complexity_pruning_path(X, yx); print("pruning path alphas", p.ccp_alphas.round(4).tolist(), "impurities", p.impurities.round(4).tolist())
print("root as a leaf R = 0.5, three-split tree R = 0; alpha_eff = (0.5 - 0)/(4 - 1) = %.4f" % (0.5 / 3))
for a in (0.1, 0.2):
    m = DTC(ccp_alpha=a, random_state=0).fit(X, yx); print("ccp_alpha = %.1f: leaves %d, accuracy %.2f" % (a, m.get_n_leaves(), m.score(X, yx)))
for md in (0.0, 1e-6, 0.01):
    m = DTC(min_impurity_decrease=md, random_state=0).fit(X, yx); print("min_impurity_decrease = %g: leaves %d, accuracy %.2f" % (md, m.get_n_leaves(), m.score(X, yx)))
