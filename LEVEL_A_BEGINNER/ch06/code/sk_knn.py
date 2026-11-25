# The same example with scikit-learn.
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

X = [[1, 1], [2, 2], [1, 3], [4, 4], [5, 2], [3.5, 1]]
c = [0, 0, 0, 1, 1, 1]
t = [12, 15, 13, 24, 27, 20]
q = [[3, 2.5]]
m = KNeighborsClassifier(n_neighbors=3).fit(X, c)
dist, idx = m.kneighbors(q)
print("neighbors:", idx[0], " distances:", dist[0].round(3))
print("class:", m.predict(q)[0], "  probabilities:",
      m.predict_proba(q)[0].round(3))
r = KNeighborsRegressor(n_neighbors=3, weights="distance").fit(X, t)
print("regression, distance weights:", r.predict(q)[0].round(3))
