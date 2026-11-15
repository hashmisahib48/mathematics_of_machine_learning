# Train and test error of two models on the delivery data.
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])    # training distances
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])   # training times
x_new = np.array([2.4, 3.6, 5.5, 6.0])     # four new deliveries
y_new = np.array([5.0, 7.0, 11.0, 12.0])

line = lambda t: 0.9 + 1.7 * t             # least squares, Ch. 9

def lookup(t):                             # copy the nearest sample
    return np.array([y[np.argmin(np.abs(x - v))] for v in t])

def mse(y_true, y_pred):                   # Eq. 6.1-1 and 6.1-2
    return np.mean((y_true - y_pred) ** 2)

for name, f in (("line", line), ("lookup", lookup)):
    tr, te = mse(y, f(x)), mse(y_new, f(x_new))
    print(f"{name:6s} train {tr:.3f}  test {te:.3f}  "
          f"gap {te - tr:.3f}")
print("line   test predictions:", np.round(line(x_new), 2))
print("lookup test predictions:", lookup(x_new))
