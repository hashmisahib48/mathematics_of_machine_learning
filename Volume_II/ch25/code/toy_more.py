# Checks for Examples 25.6-1 to 25.6-3 and the numbers of Section 25.3.
import numpy as np
income = np.array([18, 22, 36, 40, 55, 62.])
repaid = np.array([0, 0, 1, 1, 0, 1])
draws = [[0, 2, 4, 4, 5, 5], [1, 1, 3, 3, 3, 5], [0, 0, 0, 1, 2, 3]]


def gini(y):
    p = np.bincount(y, minlength=2) / max(len(y), 1)
    return 1 - (p ** 2).sum()


from sklearn.tree import DecisionTreeClassifier
X6 = income[:, None]
stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X6, repaid)
tree = DecisionTreeClassifier(random_state=0).fit(X6, repaid)
print("all six: stump splits at", stump.tree_.threshold[0],
      "; full tree has", int(tree.get_n_leaves()), "leaves, depth",
      int(tree.get_depth()), ", thresholds",
      tree.tree_.threshold[tree.tree_.feature >= 0])
print("Example 25.6-1: weighted Gini of every candidate threshold")
for m, idx in enumerate(draws):
    x, y = income[idx], repaid[idx]
    v = np.unique(x)
    row = []
    for t in (v[:-1] + v[1:]) / 2:
        g = ((x <= t).sum() * gini(y[x <= t])
             + (x > t).sum() * gini(y[x > t])) / len(y)
        row.append(f"{t:.1f}: {g:.4f}")
    print(f"  sample {m + 1}: " + "   ".join(row))

print("Example 25.6-2: variance of the mean of B equal-variance predictors")
s2, rho = 4.0, 0.25
for B in (1, 3, 12, 100):
    S = s2 * ((1 - rho) * np.eye(B) + rho * np.ones((B, B)))
    exact = np.ones(B) @ S @ np.ones(B) / B ** 2      # (1/B^2) 1' S 1
    formula = rho * s2 + (1 - rho) * s2 / B           # Eq. 25.3-8
    print(f"  B = {B:3d}: {exact:.4f} {formula:.4f}")
print("  limit B -> infinity:", rho * s2)

print("Example 25.6-3: left out, and how many trees")
for n in (6, 10, 100, 426, 10000):
    print(f"  n = {n:5d}: (1 - 1/n)^n = {(1 - 1 / n) ** n:.4f}")
print("  1/e =", round(np.exp(-1), 4), "  1 - 1/e =", round(1 - np.exp(-1), 4))
rng = np.random.default_rng(0)
n = 426
left = np.mean([len(np.setdiff1d(np.arange(n), rng.integers(0, n, n)))
                for _ in range(2000)]) / n
print("  simulated fraction left out, n = 426:", round(left, 4))
q = 1 - (1 - 1 / n) ** n                      # probability of being drawn
for B in (10, 14, 20, 30):
    print(f"  B = {B:2d}: expected samples never out of bag: "
          f"{n * q ** B:.4f}")
B_min = int(np.ceil(np.log(n) / -np.log(q)))
print("  smallest B with fewer than one:", B_min)

print("Duplicated rows equal counts as weights (stump, draw 0)")
from sklearn.tree import DecisionTreeClassifier
Xi = income.reshape(-1, 1)
d = np.array(draws[0])
a = DecisionTreeClassifier(max_depth=1).fit(Xi[d], repaid[d])
b = DecisionTreeClassifier(max_depth=1).fit(
    Xi, repaid, sample_weight=np.bincount(d, minlength=6))
print("  same threshold:", a.tree_.threshold[0] == b.tree_.threshold[0],
      a.tree_.threshold[0])
print("  same predictions:", bool((a.predict(Xi) == b.predict(Xi)).all()))
