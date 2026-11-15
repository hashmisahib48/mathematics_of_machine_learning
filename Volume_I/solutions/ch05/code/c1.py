# Problem 5C.1: three ways to decide whether a point with zero gradient is a minimum.
import numpy as np
from scipy.optimize import brentq

fs = {  # name: (f, gradient, Hessian) at the origin
 "x^2 + y^2 (minimum)":       (lambda p: p[0]**2 + p[1]**2,        lambda p: np.array([2*p[0], 2*p[1]]),        np.diag([2., 2.])),
 "x^2 - y^2 (saddle)":        (lambda p: p[0]**2 - p[1]**2,        lambda p: np.array([2*p[0], -2*p[1]]),       np.diag([2., -2.])),
 "x^4 + y^4 (flat minimum)":  (lambda p: p[0]**4 + p[1]**4,        lambda p: np.array([4*p[0]**3, 4*p[1]**3]),  np.zeros((2, 2))),
 "x^3 + y^2 (no minimum)":    (lambda p: p[0]**3 + p[1]**2,        lambda p: np.array([3*p[0]**2, 2*p[1]]),     np.diag([0., 2.])),
}
rng = np.random.default_rng(0)
D = rng.normal(size=(1000, 2)); D /= np.linalg.norm(D, axis=1)[:, None]   # 1000 random directions
print(f"{'function':26s} {'eig(H)':>12s} {'share of directions':>20s} {'GD from 0.01 away':>20s}")
for name, (f, g, H) in fs.items():
    ev = np.linalg.eigvalsh(H)
    down = np.mean([f(0.1 * d) < f(np.zeros(2)) for d in D])             # 1000 function values
    ends = []
    for d in D[:20]:                                                     # perturbed start, plain descent
        p = 0.01 * d
        for _ in range(300): p = p - 0.05 * g(p)
        ends.append(np.linalg.norm(p))
    print(f"{name:26s} {str(ev.round(1)):>12s} {down:20.3f} {np.median(ends):20.3g}")
print("(share of directions: fraction of 1000 random steps of length 0.1 that lower f; GD: median distance to the origin after 300 steps)")

print("\nthe curve of the problem figure, f(w) = sin(2w) + 0.15 w^2")
f = lambda w: np.sin(2*w) + 0.15*w**2; d1 = lambda w: 2*np.cos(2*w) + 0.3*w; d2 = lambda w: -4*np.sin(2*w) + 0.3
print(f" f'(0) = {d1(0):.3f}, f''(0) = {d2(0):.3f}   (the marked point is on a slope, not on a flat spot)")
grid = np.linspace(-3, 3, 6001); s = np.sign(d1(grid)); roots = [brentq(d1, grid[i], grid[i+1]) for i in np.where(s[:-1] != s[1:])[0]]
for r in roots: print(f" stationary w = {r:+.3f}   f = {f(r):+.3f}   f'' = {d2(r):+.2f}   {'minimum' if d2(r) > 0 else 'maximum'}")
