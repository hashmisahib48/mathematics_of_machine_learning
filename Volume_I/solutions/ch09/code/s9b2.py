# Problem 9B.2: sliding the last delivery time
import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
X = x[:, None]
print(" y5      w       b   w from formula   pred(6 km)")
for y5 in (10, 15, 20, 25, 30):
    y = np.array([3.0, 4.0, 6.0, 7.0, y5])
    m = LinearRegression().fit(X, y)
    w_formula = 1.7 + 0.2 * (y5 - 10)        # derived from Eq. 9.3-5
    print(f"{y5:3d} {m.coef_[0]:7.3f} {m.intercept_:7.3f}"
          f" {w_formula:12.3f} {m.predict([[6.0]])[0]:12.2f}")

# the same shift of +10 applied to each delivery in turn
print("\nshift of +10 at delivery i:   w      b")
y0 = np.array([3.0, 4.0, 6.0, 7.0, 10.0])
for i in range(5):
    y = y0.copy()
    y[i] += 10
    m = LinearRegression().fit(X, y)
    c = (x[i] - x.mean()) / np.sum((x - x.mean()) ** 2)
    print(f"  i = {i + 1}  (x = {x[i]:.0f})  {m.coef_[0]:12.3f}"
          f" {m.intercept_:6.3f}    weight c_i = {c:+.1f}")
