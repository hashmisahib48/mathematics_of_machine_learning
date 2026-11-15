# Toy data: three models of battery use against speed, by hand.
import numpy as np
from sklearn.preprocessing import SplineTransformer

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])    # speed, units of 5 km/h
y = np.array([2.0, 3.0, 8.0, 15.0, 22.0])  # battery used, per cent
sst = np.sum((y - y.mean()) ** 2)

def report(name, y_hat):
    sse = np.sum((y - y_hat) ** 2)
    print(f"{name:10s} SSE = {sse:6.2f}  R2 = {1 - sse/sst:.4f}")
    print("   r =", np.round(y - y_hat, 2))

u = x - x.mean()                  # centred speed
q = u ** 2 - np.mean(u ** 2)      # centred square, here u^2 - 2
c1 = (u @ y) / (u @ u)            # Eq. 15.5-2
c2 = (q @ y) / (q @ q)
print(f"c1 = {c1:.2f}  c2 = {c2:.2f}  ybar = {y.mean():.1f}")
report("line", y.mean() + c1 * u)
report("quadratic", y.mean() + c1 * u + c2 * q)

Phi = np.c_[np.ones(5), x, x ** 2]            # columns 1, x, x^2
w = np.linalg.solve(Phi.T @ Phi, Phi.T @ y)   # normal equations
print("weights of 1, x, x^2:", np.round(w, 4))

H = SplineTransformer(n_knots=3, degree=1).fit_transform(x[:, None])
v = np.linalg.solve(H.T @ H, H.T @ y)     # hats at knots 1, 3, 5
print("hat weights:", np.round(v, 4))
print("fitted:", np.round(H @ v, 4))
report("hat spline", H @ v)
