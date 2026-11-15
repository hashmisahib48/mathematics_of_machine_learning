# Section 56.4: how many candidates, how many fits.
import math
from sklearn.model_selection import ParameterGrid, ParameterSampler
from scipy.stats import loguniform

C = [1, 10, 100, 1000]
g = [1e-4, 1e-3, 1e-2, 1e-1, 1]
two = [{"kernel": ["rbf"], "C": C, "gamma": g},
       {"kernel": ["linear"], "C": C}]
flat = {"kernel": ["rbf", "linear"], "C": C, "gamma": g}
K = 5
for name, pg in (("list of two dicts", two), ("one dict", flat)):
    n = len(ParameterGrid(pg))
    print(f"{name:18s} candidates {n:3d}  fits {K * n + 1}")
lin = [p for p in ParameterGrid(flat) if p["kernel"] == "linear"]
print("linear candidates in the one dict:", len(lin),
      " distinct values of C:", len({p["C"] for p in lin}))

q, P = 0.05, 0.95                     # Eq. 56.3-3
n = math.ceil(math.log(1 - P) / math.log(1 - q))
print(f"draws for P = {P} when q = {q}: {n}, check",
      round(1 - (1 - q) ** n, 4))

# lists are sampled without replacement, distributions with it
s = list(ParameterSampler({"C": C, "gamma": g}, 6, random_state=0))
print("distinct of 6 draws from lists:",
      len({(p["C"], p["gamma"]) for p in s}))
d = ParameterSampler({"C": loguniform(1, 1e3)}, 3, random_state=0)
print("loguniform draws:", [round(float(p["C"]), 1) for p in d])
