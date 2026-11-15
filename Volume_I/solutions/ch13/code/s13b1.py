# Problem 13B.1: MyHuber of Listing 13.3 against HuberRegressor
import sys
import numpy as np
from sklearn.linear_model import HuberRegressor
sys.path.insert(0, "../../../ch13/code")    # the book's own helper
from huber_scratch import MyHuber

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([3.0, 4.0, 6.0, 7.0, 20.0])    # y5 = 20 is wrong
X = x[:, None]

print(" eps |    MyHuber         |  HuberRegressor    | scale  k")
print("     |  slope  intercept  |  slope  intercept  |")
for eps in (1.2, 1.35, 1.4, 1.5):
    me = MyHuber(epsilon=eps, alpha=0.0, n_iter=2000).fit(X, y)
    sk = HuberRegressor(epsilon=eps, alpha=0, max_iter=2000,
                        tol=1e-10).fit(X, y)
    r = y - me.intercept_ - me.coef_[0] * x     # residuals
    k = int(np.sum(np.abs(r) > eps * me.scale_))    # beyond eps*sigma
    print(f"{eps:4.2f} | {me.coef_[0]:7.4f} {me.intercept_:9.4f}  |"
          f" {sk.coef_[0]:7.4f} {sk.intercept_:9.4f}  |"
          f" {me.scale_:5.3f}  {k}")

# the least-squares line for comparison
ols = np.polyfit(x, y, 1)
print(f"\nleast squares: slope {ols[0]:.4f}, "
      f"intercept {ols[1]:.4f}")
