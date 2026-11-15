# Numbers quoted in the model answers to the questions of Chapter 1
import sys, os, runpy, io, contextlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); BOOK = HERE + '/../../../ch01/code/'
np.set_printoptions(precision=4, suppress=True)

print("== Q1  a column that is a feature and a target: the six flats")
X = np.array([[4., 1], [5, 2], [6, 2], [7, 3], [8, 3], [9, 4]]); price = np.array([95., 120, 130, 160, 170, 200])
A1 = np.column_stack([np.ones(6), X[:, 0]]); w1 = np.linalg.lstsq(A1, price, rcond=None)[0]
print("price from area: intercept, slope =", w1.round(3), " MSE =", np.mean((price - A1 @ w1) ** 2).round(3))
A2 = np.column_stack([np.ones(6), price, X[:, 1]]); w2 = np.linalg.lstsq(A2, X[:, 0], rcond=None)[0]
print("area from price and rooms: weights =", w2.round(4), " fitted:", (A2 @ w2).round(2), " MSE =", np.mean((X[:, 0] - A2 @ w2) ** 2).round(4))

print("\n== Q2  delivery data: a delay measured as time minus 1.5 min per km")
d = np.arange(1., 6); t = np.array([3., 4, 6, 7, 10]); delay = t - 1.5 * d
print("delay (min):", delay, " mean delay:", delay.mean())

print("\n== Q3  learning rate limit for other losses (Eq. 1.5-3 generalised)")
y = np.array([2., 3, 4, 5, 16]); n = len(y)
print("mean loss, constant: J'' = 2, factor 1-2*eta, limit eta < 1; best eta = 0.5:",
      "c_1 =", 0 - 0.5 * (-2 * np.mean(y - 0)))
print("sum loss (n=5): J'' = 2n =", 2 * n, " factor 1-2*n*eta; limit eta <", 1 / n, " best", 1 / (2 * n))
c = 0.0
for eta in (0.5, 0.15, 0.1):
    c = 0.0
    for _ in range(3): c = c - eta * (-2 * np.sum(y - c))
    print(f"   sum loss, eta={eta}: c_3 = {c:.4f}")
x = d                                     # slope of a line y = w x through the origin on the delivery data
print("one-knob line t = w*d: J'' = 2*mean(d^2) =", 2 * np.mean(x ** 2), " limit eta <", 1 / np.mean(x ** 2))
w = 0.0; eta = 0.1
for k in range(1, 4):
    w = w - eta * (-2 * np.mean(x * (t - w * x))); print(f"   eta=0.1, step {k}: w = {w:.3f}")
print("   the least-squares w = sum(xt)/sum(x^2) =", (x @ t) / (x @ x))
xk = 100 * x                              # same feature in cm-like units (x100)
print("same feature multiplied by 100: J'' =", 2 * np.mean(xk ** 2), " limit eta <", 1 / np.mean(xk ** 2))

print("\n== Q4  training error against test error (Listing 1.3)")
buf = io.StringIO()
with contextlib.redirect_stdout(buf): g = runpy.run_path(BOOK + 'generalise.py')
P, rm = g['Polynomial'], g['rmse']
for k in (0, 1, 3, 9):
    f = P.fit(g['x_tr'], g['y_tr'], k); print(f"degree {k}: train {rm(g['y_tr'], f(g['x_tr'])):.3f}  test {rm(g['y_te'], f(g['x_te'])):.3f}")
print("noise level 0.25; test/train ratio for degree 9:", round(7.774 / 0.207, 1), " degree 3 vs degree 9 test:", round(7.774 / 0.257, 1))
print("n_train =", len(g['x_tr']), " knobs of degree 9 =", 10)
