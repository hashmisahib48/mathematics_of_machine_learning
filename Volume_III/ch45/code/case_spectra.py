# Case study: 5-NN on the spectra, raw and projected.
import time
from spectra import make_spectra
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.random_projection import (GaussianRandomProjection,
                                       SparseRandomProjection)

X, y = make_spectra()
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
print("5-NN accuracy and fit+predict time (brute force)")
def run(name, A, B):
    t0 = time.perf_counter()
    acc = KNeighborsClassifier(5, algorithm="brute").fit(
        A, ytr).score(B, yte)
    print(f"  {name:18s} p = {A.shape[1]:5d}  accuracy {acc:.3f}"
          f"  {time.perf_counter() - t0:.3f} s")
run("raw", Xtr, Xte)
for k in (10, 30, 100):
    rp = GaussianRandomProjection(k, random_state=0).fit(Xtr)
    run(f"Gaussian, k = {k}", rp.transform(Xtr), rp.transform(Xte))
rp = SparseRandomProjection(30, random_state=0).fit(Xtr)
run("sparse, k = 30", rp.transform(Xtr), rp.transform(Xte))
