# Problem 2B.1: cosines between the mean vectors of the iris classes
import numpy as np
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
mu = np.array([X[y == k].mean(axis=0)   # class means
               for k in range(3)])
names = ["setosa", "versicolor", "virginica"]
for k in range(3):
    print(f"{names[k]:11s} mean = {np.round(mu[k], 3)}")

def cosine(a, b):               # Eq. 2.3-4
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

for i, j in [(0, 1), (0, 2), (1, 2)]:
    cs = cosine(mu[i], mu[j])
    ang = np.degrees(np.arccos(cs))
    print(f"cos({names[i]}, {names[j]}) = {cs:.4f}"
          f"   angle {ang:.2f} deg")

# The same after subtracting the overall mean from every class mean
c = mu - X.mean(axis=0)
print("centred class means, cosines:")
for i, j in [(0, 1), (0, 2), (1, 2)]:
    print(f"  {names[i]} / {names[j]}: {cosine(c[i], c[j]):.4f}")
lens = np.linalg.norm(mu, axis=1)
print("lengths of the class means:", np.round(lens, 3))
