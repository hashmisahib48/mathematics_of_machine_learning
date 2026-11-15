# The fruit data of Chapter 20: a 3-nearest-neighbour vote by hand.
import numpy as np

# columns: weight (units of 10 g), colour score; 0 apricot, 1 plum
X = np.array([[5, 3], [6, 2], [3, 4], [4, 1],
              [7, 5], [8, 8], [6, 9], [9, 8]], dtype=float)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
q = np.array([6.0, 5.0])                  # the query fruit

d2 = ((X - q) ** 2).sum(axis=1)           # Eq. 20.3-1 with p = 2
d = np.sqrt(d2)
order = np.argsort(d)                     # nearest first
print("point  class  d^2    d")
for i in order:
    print(f"{i + 1:3d}    {y[i]}    {d2[i]:5.0f}  {d[i]:.4f}")

k = 3
nn = order[:k]
votes = np.bincount(y[nn], minlength=2)   # Eq. 20.3-4, w_i = 1
print("\n3 nearest:", nn + 1, " votes (apricot, plum):", votes)
w = 1 / d[nn]                             # Eq. 20.3-5, w_i = 1/d_i
score = np.array([w[y[nn] == c].sum() for c in (0, 1)])
print("weights 1/d:", np.round(w, 4))
print("weighted score (apricot, plum):", np.round(score, 4))
name = ["apricot", "plum"]
print("uniform ->", name[votes.argmax()],
      "  distance-weighted ->", name[score.argmax()])
print("1-NN ->", name[y[order[0]]])
