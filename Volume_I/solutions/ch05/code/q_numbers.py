# Numbers quoted in the model answers to the Chapter 5 questions. Output: verify/q_numbers.out
import numpy as np

x = np.arange(1.0, 6.0); y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])
X = np.c_[np.ones(5), x]

def descend(X, eta, steps):
    w = np.zeros(X.shape[1])
    for _ in range(steps): w = w - eta * (-2 * X.T @ (y - X @ w))
    return w

print("== Q1  one parameter against two")
h1 = 2 * (x @ x); lam = np.linalg.eigvalsh(2 * X.T @ X)
print(f"slope only (no intercept): curvature {h1:.1f}, eta limit 2/h = {2/h1:.5f}")
print(f"slope + intercept: eigenvalues {lam.round(2)}, eta limit 2/lam_max = {2/lam[-1]:.5f}")
eta = 0.0172
w1 = descend(x[:, None], eta, 200)[0]; w2 = descend(X, eta, 200)
print(f"eta = {eta}: slope-only after 200 steps {w1:.4f} (exact {x@y/(x@x):.4f}); with intercept {np.abs(w2).max():.3g} (diverged)")
print("factor of the steep direction:", round(1 - eta * lam[-1], 3), "; slope-only factor:", round(1 - eta * h1, 3))
Xk = np.c_[np.ones(5), 100 * x]
print("second column x100: lam_max =", np.linalg.eigvalsh(2 * Xk.T @ Xk)[-1].round(0), " limit", (2 / np.linalg.eigvalsh(2 * Xk.T @ Xk)[-1]).round(7))

print("\n== Q2  a column multiplied by 1000 (km -> m)")
Xm = np.c_[np.ones(5), 1000 * x]
wk = np.linalg.lstsq(X, y, rcond=None)[0]; wm = np.linalg.lstsq(Xm, y, rcond=None)[0]
print("lstsq (SVD) weights km:", wk.round(6), " m:", wm.round(6), " predictions equal:", np.allclose(X @ wk, Xm @ wm))
for name, D in (("km", X), ("m", Xm)):
    ev = np.linalg.eigvalsh(2 * D.T @ D); print(f" {name}: eigenvalues {ev.round(2)}  kappa = {ev[-1]/ev[0]:.4g}")
ev = np.linalg.eigvalsh(2 * Xm.T @ Xm); eta = 1 / ev[-1]
w = np.zeros(2)
for t in range(100000): w = w - eta * (-2 * Xm.T @ (y - Xm @ w))
g = -2 * Xm.T @ (y - Xm @ w)
print(f" gradient descent, eta = 1/lam_max, 100000 steps: w = {w.round(5)}, |grad| = {np.linalg.norm(g):.3g}")
print(" the km version needs 936 steps for |grad| < 1e-6 (Listing 5.2)")

print("\n== Q3  cost of the Hessian")
for p in (10, 1000, 10**6):
    print(f" p = {p:>8d}: Hessian {8*p*p/1e9:12.6g} GB, solve ~ p^3/3 = {p**3/3:.3g} flops, L-BFGS (m=10) {2*10*p*8/1e6:.4g} MB")

print("\n== Q4  single-sample SGD, constant eta = 0.1")
rng = np.random.default_rng(0); n = 100; xs = rng.normal(size=n); ys = 2 + 3 * xs + rng.normal(0, 1.5, n)
Xs = np.c_[np.ones(n), xs]; vopt = np.linalg.lstsq(Xs, ys, rcond=None)[0]
Jf = lambda v: 0.5 * np.mean((ys - Xs @ v) ** 2); Jmin = Jf(vopt)
def run(batch, eta_fn, epochs=30, seed=0):
    rg = np.random.default_rng(seed); v = np.zeros(2); E = []; V = []; t = 0
    for e in range(epochs):
        idx = rg.permutation(n)
        for k in range(0, n, batch):
            b = idx[k:k + batch]; v = v - eta_fn(e) * (-Xs[b].T @ (ys[b] - Xs[b] @ v) / len(b)); V.append(v.copy())
        E.append(Jf(v) - Jmin)
    return np.array(E), np.array(V)
for name, bs, fn in (("full batch", 100, lambda e: 0.1), ("mini-batch 10", 10, lambda e: 0.1), ("one sample", 1, lambda e: 0.1),
                     ("one sample, decaying eta", 1, lambda e: 0.1 / (1 + 0.3 * e)), ("one sample, eta = 0.01", 1, lambda e: 0.01)):
    E, V = run(bs, fn); print(f" {name:26s} excess loss: epoch 30 {E[-1]:.5f}, mean of last 10 epochs {E[-10:].mean():.5f}")
E, V = run(1, lambda e: 0.1); avg = V[-1000:].mean(axis=0)
print(f" iterates of the last 10 epochs: spread (std) {V[-1000:].std(axis=0).round(3)}; their average {avg.round(3)} -> excess {Jf(avg) - Jmin:.5f}; minimum {vopt.round(3)}")

print("\n== Q5  lambda and t depend on the data")
for c in ((3., 4.), (6., 8.)):
    nc = np.hypot(*c); print(f" c = {c}: |c| = {nc:.0f}; t at lambda = 1: {nc/2:.2f}; lambda at t = 1: {nc/1 - 1:.0f}")
for name, xx, yy in (("data A (x as given)", np.array([-2., -1, 1, 2]), np.array([-3., -1, 1, 3])), ("data B (x x 10)", 10 * np.array([-2., -1, 1, 2]), np.array([-3., -1, 1, 3]))):
    w = (xx @ yy) / (xx @ xx + 1.0); print(f" ridge {name}: Sxx = {xx@xx:.0f}, Sxy = {xx@yy:.0f}, alpha = 1 -> slope {w:.4f} (OLS {xx@yy/(xx@xx):.4f})")
for name, xx in (("A", np.array([-2., -1, 1, 2])), ("B", 10 * np.array([-2., -1, 1, 2]))):
    print(f" data {name}: alpha = 1 keeps the fraction {(xx@xx)/(xx@xx+1):.4f} of the OLS slope")
