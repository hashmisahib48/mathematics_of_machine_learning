# k-NN regression on the diabetes data: the effect of k.
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          random_state=0)
for k in (1, 5, 10, 20, 40, 80):
    m = make_pipeline(StandardScaler(), KNeighborsRegressor(k))
    m.fit(X_tr, y_tr)
    print(f"k = {k:2d}: train R2 {m.score(X_tr, y_tr):.3f}"
          f"  test R2 {m.score(X_te, y_te):.3f}")
