# Problem 39B.2: SpectralCoclustering with a wrong and the right K
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralCoclustering
from sklearn.datasets import make_biclusters
from sklearn.metrics import consensus_score

# 80 x 80 table, 3 hidden blocks, noise 5, then shuffled
data, rows, cols = make_biclusters((80, 80), 3, noise=5,
                                   shuffle=False, random_state=0)
pr = np.random.default_rng(1).permutation(80)     # row permutation
pc = np.random.default_rng(2).permutation(80)     # column permutation
table = data[pr][:, pc]
tr = rows[:, pr].astype(int)                  # true blocks
tc = cols[:, pc].astype(int)
print("true blocks: rows", tr.sum(1).tolist(),
      " columns", tc.sum(1).tolist())

def jaccard(r1, c1, r2, c2):         # overlap of two sets of cells
    both = (r1 @ r2) * (c1 @ c2)
    return both / (r1.sum() * c1.sum() + r2.sum() * c2.sum() - both)

for K in (2, 3, 4, 5):
    co = SpectralCoclustering(K, random_state=0).fit(table)
    fr, fc = co.rows_.astype(int), co.columns_.astype(int)
    J = np.array([[jaccard(fr[i], fc[i], tr[j], tc[j])
                   for j in range(3)] for i in range(K)])
    i, j = linear_sum_assignment(-J)           # best matching
    truth = (rows[:, pr], cols[:, pc])
    score = consensus_score(co.biclusters_, truth)
    print(f"K = {K}: consensus {score:.3f} = {J[i, j].sum():.4f}"
          f" / {max(K, 3)}  Jaccard"
          f" {J[i, j].round(3).tolist()}")
    print("   rows per bicluster", fr.sum(1).tolist(),
          " columns", fc.sum(1).tolist())
    print("   closest true block of each:", J.argmax(1).tolist())
