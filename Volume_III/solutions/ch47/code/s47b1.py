# Problem 47B.1: leave-one-out choice of h, by the identity of
# Section 47.4 and by refitting; 10-fold search for comparison.
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KernelDensity


def waits(n, seed):                 # the 200 waits of select_h.py
    rng = np.random.default_rng(seed)
    short = rng.random(n) < 0.35
    t = np.where(short, rng.normal(54, 5, n),
                 rng.normal(80, 6, n))
    return t[:, None]

X = waits(200, 1); n = len(X)
grid = np.arange(1.0, 10.01, 0.25)
loo = []                            # the identity, one fit per h
for h in grid:
    kde = KernelDensity(bandwidth=h).fit(X)
    f = np.exp(kde.score_samples(X))
    k0 = 1 / (h * np.sqrt(2 * np.pi))       # K(0)/h, Gaussian
    loo.append(np.log((n * f - k0) / (n - 1)).mean())
loo = np.array(loo); i = loo.argmax()
print(f"identity   : h = {grid[i]:.2f}  LOO ln f = {loo[i]:.4f}")

# GridSearchCV scores a fold by the SUM of ln f over its rows:
# divide by the fold size (1 or 20) for a mean per wait.
for name, cv, m in [("LeaveOneOut", LeaveOneOut(), 1),
                    ("10-fold    ", KFold(10, shuffle=True,
                                          random_state=0), 20)]:
    gs = GridSearchCV(KernelDensity(), {"bandwidth": grid}, cv=cv)
    per = gs.fit(X).cv_results_["mean_test_score"] / m
    print(f"{name}: h = {grid[per.argmax()]:.2f}  "
          f"ln f = {per.max():.4f}")
    if m == 1:
        d = abs(per - loo).max()
        print(f"largest difference: {d:.1e}")
for h in (1.0, 2.0, 2.5, 3.0, 5.0, 10.0):   # the curve
    print(f"h = {h:5.2f}  LOO ln f = {loo[grid == h][0]:.4f}")
