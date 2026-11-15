# The derivative as a limit; rules and functions of learning.
import numpy as np

f = lambda x: x ** 2
for h in [1.0, 0.1, 0.01, 0.001]:          # Eq. 3.1-1 with x = 1
    print(f"h = {h:<6} (f(1+h) - f(1)) / h = {(f(1+h)-f(1))/h:.4f}")

s = lambda z: 1 / (1 + np.exp(-z))         # sigmoid, Eq. 3.2-1
table = [            # name, function, derivative, point
    ("power",    "x**5",          "5*x**4",                0.5),
    ("sum",      "x**2 + 3*x",    "2*x + 3",               0.5),
    ("product",  "x*np.exp(x)",   "np.exp(x)*(1 + x)",     0.5),
    ("quotient", "x/(1 + x**2)",  "(1 - x**2)/(1 + x**2)**2", 0.5),
    ("chain",    "(3*x + 1)**4",  "12*(3*x + 1)**3",       0.5),
    ("sigmoid",  "s(x)",          "s(x)*(1 - s(x))",       0.7),
    ("softplus", "np.log1p(np.exp(x))", "s(x)",            0.7),
    ("abs",      "abs(x)",        "np.sign(x)",            0.7),
]
print("\nname       by the rule   by numbers   at x")
for name, fs, ds, x0 in table:
    g = lambda x: eval(fs)
    h = 1e-6                               # central difference
    num = (g(x0 + h) - g(x0 - h)) / (2 * h)
    print(f"{name:<9} {eval(ds, globals(), {'x': x0}):12.6f}"
          f" {num:12.6f}   {x0}")

print("\nsigmoid and its slope")
for z in [-6, -2, 0, 2, 6]:
    d = s(z) * (1 - s(z))
    print(f"z = {z:+d}  sigma = {s(z):.4f}  slope = {d:.4f}")
h = 1e-6
print("\nabs at 0, slope from the left: ", (0 - abs(-h)) / h)
print("abs at 0, slope from the right:", (abs(h) - 0) / h)
