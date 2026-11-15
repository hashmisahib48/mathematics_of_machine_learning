# Toy check of Chapter 58.
import numpy as np

p = np.array([0.92, 0.75, 0.55, 0.40, 0.30, 0.18, 0.08, 0.03])
y = np.array([1, 1, 0, 1, 0, 1, 0, 0])       # Example 58.5-1
c_fn, c_fp = 5, 1
print("t* =", round(c_fp / (c_fp + c_fn), 4))   # Eq. 58.3-2
for t in (0.5, 0.18, 0.08, 0.0):
    yes = p >= t
    fn, fp = (y & ~yes).sum(), ((1 - y) & yes).sum()
    cost = c_fn * fn + c_fp * fp
    print(f"t = {t:4.2f}  FN {fn}  FP {fp}  cost {cost}")
