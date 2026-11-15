# Problem 20C.3: operations and time for one prediction, n = 10^6, p = 50
import time, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

n, p = 10 ** 6, 50
print("brute-force 5-NN: squared distances need n*p = %.1e multiplications (and as many subtractions and additions)" % (n * p))
print("  plus a selection of the 5 smallest of n numbers: about n = %.0e comparisons" % n)
print("logistic regression: w^T x + b needs p = %d multiplications" % p)
print("  ratio: %.0f; memory read per query: %.0f MB of float64 against %d bytes of weights" % (n, n * p * 8 / 1e6, p * 8))
rng = np.random.default_rng(0)
X = rng.normal(size=(n, p)).astype(np.float32); q = rng.normal(size=p).astype(np.float32)
ts = []
for _ in range(5):
    t0 = time.perf_counter(); d2 = ((X - q) ** 2).sum(1); idx = np.argpartition(d2, 5)[:5]; ts.append(time.perf_counter() - t0)
print("measured: one NumPy brute-force query (float32), best of 5: %.0f ms (the machine, not the count, sets the figure)" % (1e3 * min(ts)))
lr = LogisticRegression().fit(X[:2000], (X[:2000, 0] > 0).astype(int)); xq = q[None]
t0 = time.perf_counter()
for _ in range(2000): lr.predict(xq)
print("measured: LogisticRegression.predict for one row %.0f microseconds, almost all of it call overhead" % (1e6 * (time.perf_counter() - t0) / 2000))
w = lr.coef_[0]; t0 = time.perf_counter()
for _ in range(100000): s = w @ q
print("measured: the dot product itself %.2f microseconds" % (1e6 * (time.perf_counter() - t0) / 1e5))
print("a 1 ms budget at 10^9 multiplications per second allows %.0e multiplications: k-NN needs %.0f times that" % (1e-3 * 1e9, n * p / 1e6))
