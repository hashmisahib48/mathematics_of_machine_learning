# Problem 41B.2: random labellings and the four scores
import numpy as np
from scipy.stats import entropy
from sklearn import metrics as m

truth = np.repeat([0, 1, 2], 50)                 # three classes of 50
rng = np.random.default_rng(0)
score = {"RI": m.rand_score, "ARI": m.adjusted_rand_score,
         "NMI": m.normalized_mutual_info_score,
         "AMI": m.adjusted_mutual_info_score}

print("30 random clusters, mean of 20 draws")
labs = [rng.integers(0, 30, 150) for _ in range(20)]
for name, f in score.items():
    print(f"{name:4s}{np.mean([f(truth, l) for l in labs]):8.3f}")

Hc = entropy(np.bincount(truth))                 # nats
Hu = np.mean([entropy(np.bincount(l)) for l in labs])
print(f"H(classes) = {Hc:.3f}   mean H(clusters) = {Hu:.3f}")

# why: the expected value of the Rand index from pair counts
n, k = 150, 30
pt = 3 * 1225 / (n * (n - 1) / 2)    # together in classes
pc = 1 / k                           # together in clusters
print(f"\nexpected RI = {pt*pc + (1-pt)*(1-pc):.3f}")

# the same for other numbers of clusters
print("\n k      RI    ARI    NMI    AMI    MI   2(k-1)/2n")
for k in (3, 10, 30, 100):
    rng = np.random.default_rng(0); rows = []
    for _ in range(20):
        l = rng.integers(0, k, n)
        rows.append([f(truth, l) for f in score.values()]
                    + [m.mutual_info_score(truth, l)])
    r = np.mean(rows, axis=0)
    print(f"{k:3d}{r[0]:8.3f}{r[1]:7.3f}{r[2]:7.3f}{r[3]:7.3f}"
          f"{r[4]:7.3f}{2*(k-1)/(2*n):9.3f}")
