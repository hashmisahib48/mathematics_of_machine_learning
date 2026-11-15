# Problem 24B.2: five noise columns appended to the breast-cancer data
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DTC

data = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data, data.target, test_size=0.25, random_state=0,
    stratify=data.target)
rng = np.random.default_rng(0)
noise_tr = rng.normal(size=(len(X_tr), 5))
noise_te = rng.normal(size=(len(X_te), 5))

base = DTC(random_state=0).fit(X_tr, y_tr)
tree = DTC(random_state=0).fit(np.c_[X_tr, noise_tr], y_tr)
imp = tree.feature_importances_
print("leaves: %d without noise, %d with" % (
    base.get_n_leaves(), tree.get_n_leaves()))
print("test accuracy: %.4f without, %.4f with" % (
    base.score(X_te, y_te), tree.score(np.c_[X_te, noise_te], y_te)))
print("noise importances:", *["%.4f" % v for v in imp[30:]])
print("total noise importance %.4f" % imp[30:].sum())

t, n = tree.tree_, tree.tree_.n_node_samples
for k in np.where(t.feature >= 30)[0]:      # nodes that asked noise
    a, b = t.children_left[k], t.children_right[k]
    gain = (n[k] * t.impurity[k] - n[a] * t.impurity[a]
            - n[b] * t.impurity[b]) / n[0]
    print("node %d, column %d: %d samples -> %d + %d, gain %.4f" % (
        k, t.feature[k], n[k], n[a], n[b], gain))
print("gain of the whole tree = root impurity %.4f" % t.impurity[0])

tot = []                                    # 20 other noise draws
for s in range(1, 21):
    nz = np.random.default_rng(s).normal(size=(len(X_tr), 5))
    f = DTC(random_state=0).fit(np.c_[X_tr, nz], y_tr)
    tot.append(f.feature_importances_[30:].sum())
print("20 other draws: mean %.4f, min %.4f, max %.4f" % (
    np.mean(tot), min(tot), max(tot)))
