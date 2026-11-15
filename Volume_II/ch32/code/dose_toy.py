# Toy A: pool adjacent violators with the merges printed.
import numpy as np
from sklearn.isotonic import isotonic_regression

y = np.array([2., 5., 3., 4., 1., 6., 5., 7.])
blocks = []                          # each block holds [sum, count]
mean = lambda b: b[0] / b[1]
for k, v in enumerate(y, start=1):
    blocks.append([v, 1])
    print(f"dose {k}: push {v:g}")
    while len(blocks) > 1 and mean(blocks[-2]) > mean(blocks[-1]):
        s, c = blocks.pop()          # merge the last two blocks
        blocks[-1][0] += s
        blocks[-1][1] += c
        print(f"    merge: mean {mean(blocks[-1]):.4f} "
              f"over {blocks[-1][1]} doses")
fit = np.concatenate([[s / c] * c for s, c in blocks])
ref = isotonic_regression(y)
print("pooled fit :", fit)
print("sklearn    :", ref)
print("sum y, sum fit:", y.sum(), fit.sum())
print("largest difference:", np.abs(fit - ref).max())
