# Expected training and test error of least squares (p = 4).
import numpy as np

rng = np.random.default_rng(62)
sigma, p, reps = 0.25, 4, 20000

print("   n  train sim  sigma^2(1-p/n)  test sim  sigma^2(1+p/n)")
for n in (8, 20, 40, 80, 160):
    x = (np.arange(n) + 0.5) / n                  # fixed design
    X = np.vander(x, p, increasing=True)          # 1, x, x^2, x^3
    H = X @ np.linalg.pinv(X)                     # hat matrix
    mean = 1 + 4 * x - 10 * x**2 + 6 * x**3       # a true cubic
    E1 = rng.normal(0, sigma, (reps, n))          # training noise
    E2 = rng.normal(0, sigma, (reps, n))          # fresh noise
    Y = mean + E1
    Yhat = Y @ H.T                                # fitted values
    train = np.mean((Y - Yhat) ** 2)              # Eq. 6.4-2
    test = np.mean((mean + E2 - Yhat) ** 2)       # Eq. 6.4-3
    print(f"{n:4d} {train:10.4f} {sigma**2*(1-p/n):15.4f}"
          f" {test:9.4f} {sigma**2*(1+p/n):15.4f}")
print("trace of H for n = 160:", round(np.trace(H), 6))
