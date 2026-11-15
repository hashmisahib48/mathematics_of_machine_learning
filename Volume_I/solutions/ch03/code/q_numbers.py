# Numbers for the model answers to the questions for discussion, Chapter 3
import numpy as np
sg = lambda z: 1 / (1 + np.exp(-z))
x = np.array([1., 2, 3, 4, 5]); y = np.array([3., 4, 6, 7, 10]); X = np.c_[np.ones(5), x]
gJ = lambda v, X=X: -2 * X.T @ (y - X @ v)

print("== Q1  sign and size of a derivative")
g = gJ(np.array([0., 1.])); print("dJ/dw at (0,1) (x in km):", g[1])
Xm = np.c_[np.ones(5), 1000 * x]                 # distance in metres: w_m = w/1000
gm = gJ(np.array([0., 0.001]), Xm); print("dJ/dw_m at the same line (x in m):", gm[1], " ratio:", gm[1] / g[1])
print("step eta = 0.01 changes w by", -0.01 * g[1], " in km units; the same eta in metres changes w_m by", -0.01 * gm[1])

print("\n== Q2  gradient versus direction of the minimum, delivery loss at (3, 1)")
v = np.array([3., 1.]); H = 2 * X.T @ X; vs = np.linalg.solve(X.T @ X, X.T @ y)
g = gJ(v); d = vs - v
print("minimum:", vs, " gradient at (3,1):", g, " direction to minimum:", d)
c = (-g) @ d / (np.linalg.norm(g) * np.linalg.norm(d)); print("cos =", c, " angle =", np.degrees(np.arccos(c)), "deg")
print("Newton direction -H^-1 g:", -np.linalg.solve(H, g))
ev = np.linalg.eigvalsh(H); print("Hessian eigenvalues:", ev, " ratio:", ev[1] / ev[0])
print("circular bowl f = u^2+v^2 at (3,1): -grad =", -2 * np.array([3., 1]), " parallel to the minimiser direction (-3,-1)")

print("\n== Q3  saddle")
f = lambda u, v: u ** 2 - v ** 2
print("f(0,0) =", f(0, 0), " f(0.1,0) =", f(0.1, 0), " f(0,0.1) =", f(0, 0.1), " Hessian eig: 2, -2")
print("delivery Hessian eigenvalues:", ev, " min over unit u of u'Hu is", ev[0])
rng = np.random.default_rng(0); Xr = rng.normal(size=(30, 4)); print("random X: eig of 2X'X =", np.linalg.eigvalsh(2 * Xr.T @ Xr).round(3))

print("\n== Q4  ten sigmoids")
print("0.25^10 =", 0.25 ** 10, " 0.25^5 =", 0.25 ** 5)
z = 0.0; prod = 1.0
for k in range(10):
    s = sg(z); prod *= s * (1 - s); z = s
print("chain z -> sigma(z), ten times, starting at 0 (all weights 1): product of slopes =", prod)
for w in (1, 2, 4, 8):
    z = 0.5; prod = 1.0
    for k in range(10):
        s = sg(w * z); prod *= w * s * (1 - s); z = s
    print(f"weight {w}: derivative of the chain at start z = 0.5: {prod:.3e}")
print("weight needed per layer to cancel 0.25:", 1 / 0.25)
