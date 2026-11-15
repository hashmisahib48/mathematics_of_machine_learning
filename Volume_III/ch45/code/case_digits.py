# Case study: digits, an approximate feature map and a linear model.
import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, LinearSVC

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X / 16, y, test_size=0.25, random_state=0, stratify=y)
g = 1 / (Xtr.shape[1] * Xtr.var())          # SVC(gamma="scale")
print(f"gamma = {g:.4f}  train {Xtr.shape}  test {Xte.shape}")

def run(name, model):
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    t = time.perf_counter() - t0
    print(f"{name:28s} test accuracy {model.score(Xte, yte):.4f}"
          f"  fit {t:.3f} s")

run("LinearSVC on the pixels", LinearSVC(max_iter=10000))
run("SVC, RBF (exact)", SVC(gamma=g))
for D in (50, 200, 800):
    for cls in (RBFSampler, Nystroem):
        run(f"{cls.__name__}({D}) + LinearSVC", make_pipeline(
            cls(gamma=g, n_components=D, random_state=0),
            LinearSVC(max_iter=10000)))
