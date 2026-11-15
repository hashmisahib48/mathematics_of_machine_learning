# Voting and stacking regressors on the diabetes split of Chapter 9.
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      random_state=0)
est = [("ols", LinearRegression()),
       ("forest", RandomForestRegressor(
           n_estimators=300, max_features=0.33, min_samples_leaf=5,
           random_state=0)),
       ("boost", GradientBoostingRegressor(random_state=0))]

def report(name, model):
    mse = mean_squared_error(yte, model.fit(Xtr, ytr).predict(Xte))
    print(f"{name:14s} test R2 {model.score(Xte, yte):.4f}"
          f"  RMSE {mse ** 0.5:.1f}")

for name, m in est:
    report(name, m)
vote = VotingRegressor(est)
report("average", vote)
stack = StackingRegressor(est)         # final_estimator=RidgeCV()
report("stack", stack)
print("stack weights", stack.final_estimator_.coef_.round(3),
      " intercept", round(stack.final_estimator_.intercept_, 1),
      " alpha", stack.final_estimator_.alpha_)

pos = StackingRegressor(est, final_estimator=LinearRegression(
    positive=True))                    # Breiman: weights not negative
report("stack, w >= 0", pos)
print("positive stack weights", pos.final_estimator_.coef_.round(3),
      " sum", round(pos.final_estimator_.coef_.sum(), 3))

F = vote.transform(Xte)                # one column per member
fbar = F.mean(1)                       # Eq. 29.3-6, equal weights
lhs = np.mean((fbar - yte) ** 2)
err = np.mean((F - yte[:, None]) ** 2)
spread = np.mean((F - fbar[:, None]) ** 2)
print(f"MSE of the average {lhs:.1f} = mean member MSE {err:.1f}"
      f" - spread {spread:.1f}")
assert np.isclose(lhs, err - spread)
