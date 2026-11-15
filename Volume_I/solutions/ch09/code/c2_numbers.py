# Problem 9C.2: how R^2 behaves when p approaches n (averages over many data sets). Run from code/.
import numpy as np
from sklearn.linear_model import LinearRegression
n, sig, reps, nte = 60, 2.0, 400, 3000
var_y = 5 + sig ** 2                      # y = 2 x1 - x2 + noise, features N(0,1)
rng = np.random.default_rng(7)
ps = list(range(2, 59, 4))
res = {p: ([], []) for p in ps}
for _ in range(reps):
    X = rng.normal(size=(n, 58)); y = 2 * X[:, 0] - X[:, 1] + rng.normal(0, sig, n)
    Xt = rng.normal(size=(nte, 58)); yt = 2 * Xt[:, 0] - Xt[:, 1] + rng.normal(0, sig, nte)
    for p in ps:
        m = LinearRegression().fit(X[:, :p], y)
        res[p][0].append(m.score(X[:, :p], y)); res[p][1].append(m.score(Xt[:, :p], yt))
print("   p   mean train   formula   median test   mean test   formula (n-p-2>0)")
out = []
for p in ps:
    tr, te = np.array(res[p][0]), np.array(res[p][1])
    f_tr = 1 - sig ** 2 * (n - p - 1) / ((n - 1) * var_y)
    f_te = 1 - sig ** 2 * (1 + p / (n - p - 2)) / var_y if n - p - 2 > 0 else float("nan")
    print(f"{p:4d} {tr.mean():10.3f} {f_tr:9.3f} {np.median(te):12.3f} {te.mean():11.3f} {f_te:12.3f}")
    out.append((p, tr.mean(), f_tr, np.median(te)))
print("\nadjusted R^2 of the report: 1 - (1 - 0.97)(n - 1)/(n - p - 1) =", round(1 - (1 - 0.97) * (n - 1) / (n - 58 - 1), 2))
print("residual degrees of freedom n - p - 1 at p = 58:", n - 58 - 1)
np.save("../verify/c2_curve.npy", np.array(out))
