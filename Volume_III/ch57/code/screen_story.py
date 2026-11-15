# Section 57.1: accuracy of a screening test for a 5 % disease.
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score,
                             balanced_accuracy_score, recall_score)
from screendata import make_screen

s, y = make_screen(n=2000, prevalence=0.05, d=2.0, seed=0)
rules = {"nobody is ill": np.zeros_like(y),
         "screen, score > 1.5": (s > 1.5).astype(int)}
print(f"{int(y.sum())} ill among {len(y)} people")
print(f"{'rule':22s}{'acc':>7s}{'bal.acc':>9s}{'recall':>8s}"
      f"{'precision':>11s}")
for name, p in rules.items():
    print(f"{name:22s}{accuracy_score(y, p):7.3f}"
          f"{balanced_accuracy_score(y, p):9.3f}"
          f"{recall_score(y, p):8.3f}"
          f"{precision_score(y, p, zero_division=0):11.3f}")
