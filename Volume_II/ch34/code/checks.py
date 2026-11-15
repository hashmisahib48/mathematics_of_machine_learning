# Numbers quoted in the text of Chapter 34 that no listing prints.
import warnings
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.semi_supervised import (LabelPropagation, LabelSpreading,
                                     SelfTrainingClassifier)
from sklearn.svm import SVC
from sklearn.utils.fixes import laplacian

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
lab, _ = train_test_split(np.arange(len(ytr)), train_size=30,
                          random_state=0, stratify=ytr)
yy = -np.ones(len(ytr), int)
yy[lab] = ytr[lab]
print("1. graph used by the knn kernel (n_neighbors=7)")
nn = NearestNeighbors(n_neighbors=7).fit(Xtr)
A = nn.kneighbors_graph(Xtr, 7, mode="connectivity").toarray()
A0 = A - np.eye(len(A))                  # edges to other samples
print("   diagonal ones:", int(A.diagonal().sum()), "of", len(A),
      " row sums:", {int(v) for v in A.sum(axis=1)})
print("   symmetric:", bool((A == A.T).all()),
      " share of mutual edges:",
      round(float((A0 * A0.T).sum() / A0.sum()), 3))
lp = LabelPropagation(kernel="knn", n_neighbors=7)
lp.X_ = Xtr
T = lp._build_graph().toarray()
print("   T = row-normalised A (max difference):",
      abs(T - A / A.sum(axis=1, keepdims=True)).max())
ls = LabelSpreading(kernel="knn", n_neighbors=7)
ls.X_ = Xtr
S = ls._build_graph().toarray()
c = A0.sum(axis=0)                       # in-degrees without self
print("   points that nobody has among their neighbours:",
      int((c == 0).sum()))
c[c == 0] = 1                            # scipy: isolated nodes get 1
S2 = A0 / np.sqrt(np.outer(c, c))
print("   S = A0 / sqrt(c_i c_j), c = column sums of A minus I:",
      abs(S - S2).max(), " S symmetric:", bool(np.allclose(S, S.T)))
W = rbf_kernel(Xtr[:200], gamma=0.02)
ls = LabelSpreading(kernel="rbf", gamma=0.02)
ls.X_ = Xtr[:200]
S = ls._build_graph()
W0 = W - np.eye(200)
d0 = W0.sum(axis=1)
print("   rbf: S = W0/sqrt(d_i d_j), W0 = W minus I:",
      abs(S - W0 / np.sqrt(np.outer(d0, d0))).max())

print("2. ten wrong labels among 100: does soft clamping repair them?")
lab100, _ = train_test_split(np.arange(len(ytr)), train_size=100,
                             random_state=0, stratify=ytr)
rng = np.random.default_rng(1)
bad = rng.choice(lab100, 10, replace=False)
yn = ytr.copy()
yn[bad] = (yn[bad] + rng.integers(1, 10, 10)) % 10
y100 = -np.ones(len(ytr), int)
y100[lab100] = yn[lab100]
U = y100 == -1
good = np.setdiff1d(lab100, bad)
models = [("propagation", LabelPropagation(kernel="knn", max_iter=1000))]
models += [(f"spreading alpha={a}", LabelSpreading(
    kernel="knn", alpha=a, max_iter=300)) for a in (0.2, 0.8, 0.99)]
for name, m in models:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(Xtr, y100)
    t = m.transduction_
    rep = int((t[bad] == ytr[bad]).sum())
    lost = int((t[good] != ytr[good]).sum())
    print(f"   {name:20s} wrong labels repaired {rep:2d}  right "
          f"labels lost {lost}  accuracy on unlabelled "
          f"{np.mean(t[U] == ytr[U]):.3f}")

print("3. the base estimator needs predict_proba")
for est in (SVC(), None):
    try:
        SelfTrainingClassifier(est).fit(
            Xtr[:50], np.r_[ytr[:20], -np.ones(30, int)])
    except Exception as e:
        print("  ", type(e).__name__, str(e)[:90].replace("\n", " "))
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    SVC(probability=True).fit(Xtr[:60], ytr[:60])
print("   SVC(probability=True) warns:", [x.category.__name__ for x in w])

print("4. scaling the pixels before the graph (30 labels, 5 draws)")
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(Xtr)
res = {"raw": [], "standardised": []}
for seed in range(5):
    l5, _ = train_test_split(np.arange(len(ytr)), train_size=30,
                             random_state=seed, stratify=ytr)
    y5 = -np.ones(len(ytr), int)
    y5[l5] = ytr[l5]
    for name, (A_, B_) in {"raw": (Xtr, Xte),
                           "standardised": (sc.transform(Xtr),
                                            sc.transform(Xte))}.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = LabelSpreading(kernel="knn", alpha=0.8).fit(A_, y5)
            res[name].append(m.score(B_, yte))
for name, v in res.items():
    print(f"   {name:13s} test accuracy {np.mean(v):.3f} +- {np.std(v):.3f}")
