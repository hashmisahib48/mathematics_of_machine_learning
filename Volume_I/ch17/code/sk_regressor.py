# SGDRegressor with three losses, on the delivery data (§17.10).
import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor

x = np.arange(1.0, 6.0).reshape(-1, 1)           # distances, km
data = {"clean": np.array([3.0, 4, 6, 7, 10]),
        "y5 = 20": np.array([3.0, 4, 6, 7, 20])}  # one wrong entry

def sgd(loss, eps=0.1):
    return SGDRegressor(loss=loss, epsilon=eps, penalty=None,
                        eta0=0.01, max_iter=5000, tol=None,
                        random_state=0)

models = {"OLS (exact)": LinearRegression(),
          "squared_error": sgd("squared_error"),
          "huber, eps = 1": sgd("huber", 1.0),
          "epsilon_insens.": sgd("epsilon_insensitive", 0.1)}
print("targets   model             slope  intercept")
for tag, y in data.items():
    for name, m in models.items():
        m.fit(x, y)
        b = np.ravel(m.intercept_)[0]       # float or array
        print(f"{tag:9s} {name:16s} {m.coef_[0]:6.3f}{b:10.3f}")
