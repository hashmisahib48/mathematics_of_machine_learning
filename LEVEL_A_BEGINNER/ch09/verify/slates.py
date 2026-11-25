"""Numbers printed on the slates of Chapter 9 (candidate x1 <= 1.5 of Example 9.4-1)."""
import sys
sys.path.insert(0, '../code')
from tree import gini
X = [[1, 4], [2, 7], [3, 2], [5, 6], [6, 2], [6, 8], [7, 5], [8, 4]]
y = [0, 0, 0, 1, 0, 1, 1, 0]
L = [t for x, t in zip(X, y) if x[0] <= 1.5]
R = [t for x, t in zip(X, y) if x[0] > 1.5]
print("left", len(L), "counts", (L.count(0), L.count(1)), "G_L", round(gini(L), 4))
print("right", len(R), "counts", (R.count(0), R.count(1)), "G_R", round(gini(R), 4))
print("I_split", round((len(L) * gini(L) + len(R) * gini(R)) / 8, 4))
