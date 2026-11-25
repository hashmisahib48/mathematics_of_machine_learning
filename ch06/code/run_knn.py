# The worked example of Section 6.4 with our classes.
from knn import KNeighborsClassifier, KNeighborsRegressor

X = [[1, 1], [2, 2], [1, 3], [4, 4], [5, 2], [3.5, 1]]   # A..F
c = [0, 0, 0, 1, 1, 1]                  # classes
t = [12, 15, 13, 24, 27, 20]            # numerical targets
q = [[3, 2.5]]
for k in (1, 3, 5):
    m = KNeighborsClassifier(n_neighbors=k).fit(X, c)
    print(f"k = {k}: class {m.predict(q)[0]}")
r = KNeighborsRegressor(n_neighbors=3).fit(X, t)
print("regression, k = 3, uniform: ", round(r.predict(q)[0], 3))
r = KNeighborsRegressor(n_neighbors=3, weights="distance").fit(X, t)
print("regression, k = 3, distance:", round(r.predict(q)[0], 3))
