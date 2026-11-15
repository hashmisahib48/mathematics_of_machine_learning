# What algorithm='auto' chooses (read from the private _fit_method).
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNC

rng = np.random.default_rng(0)
y = rng.integers(0, 2, 1000)
cases = [
    ("3 features",         3, dict()),
    ("15 features",       15, dict()),
    ("16 features",       16, dict()),
    ("k = 499",            3, dict(n_neighbors=499)),
    ("k = 500",            3, dict(n_neighbors=500)),
    ("chebyshev",          3, dict(metric="chebyshev")),
    ("cosine",             3, dict(metric="cosine")),
    ("haversine",          2, dict(metric="haversine")),
    ("weighted minkowski", 3, dict(metric_params={"w": [1, 2, 3]})),
]
print("n = 1000, k = 5 unless stated; algorithm='auto'")
for name, p, kw in cases:
    m = KNC(**kw).fit(rng.uniform(0, 1, (1000, p)), y)
    print(f"{name:20s} -> {m._fit_method}")
