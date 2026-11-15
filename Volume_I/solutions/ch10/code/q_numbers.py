# Numbers quoted in the model answers to the questions of Chapter 10
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
np.set_printoptions(precision=4, suppress=True)
x1 = np.array([1, 2, 3, 4, 5.]); x2 = np.array([1, 2, 3, 5, 4.]); y = np.array([3, 4, 6, 7, 10.])
X = np.c_[x1, x2]; n = 5
def loo(X, y, a):                                   # Eq. 10.9-4, 10.9-5
    Xc, yc = X - X.mean(0), y - y.mean()
    S = np.linalg.inv(Xc.T @ Xc + a * np.eye(X.shape[1]))
    h = 1 / len(y) + np.einsum("ij,jk,ik->i", Xc, S, Xc)
    r = yc - Xc @ (S @ Xc.T @ yc)
    return np.mean((r / (1 - h)) ** 2), r @ r

print("== Q1  training SSE and leave-one-out error, two-reading toy data")
for a in (0, 0.1, 1, 3):
    e, sse = loo(X, y, a); print(f"alpha = {a:4}: training SSE = {sse:.3f}   leave-one-out mean square error = {e:.3f}")
ref = []
for i in range(n):                                  # explicit refits for alpha = 0 and 1
    k = np.arange(n) != i
    ref.append([y[i] - LinearRegression().fit(X[k], y[k]).predict(X[i:i + 1])[0], y[i] - Ridge(alpha=1).fit(X[k], y[k]).predict(X[i:i + 1])[0]])
ref = np.array(ref); print("explicit refits, mean square error (alpha = 0, alpha = 1):", (ref ** 2).mean(0).round(3))

print("\n== Q2  unit of the target, with and without a penalty on the intercept (delivery data, alpha = 10)")
xd = np.array([1, 2, 3, 4, 5.]); yd = np.array([3, 4, 6, 7, 10.]); a = 10.0
X1 = np.c_[np.ones(5), xd]
for name, f in (("minutes", 1), ("seconds", 60)):
    ys = f * yd
    m = Ridge(alpha=a).fit(xd[:, None], ys)
    wb = np.linalg.solve(X1.T @ X1 + a * np.eye(2), X1.T @ ys)
    ls = np.polyfit(xd, ys, 1)
    print(f"{name:7s}: ridge slope {m.coef_[0]:8.4f} (ratio to least squares {m.coef_[0] / ls[0]:.3f}) b = {m.intercept_:8.3f} |"
          f" intercept penalised: slope {wb[1]:8.4f} (ratio {wb[1] / ls[0]:.3f}) b = {wb[0]:8.3f}")
wb_m = np.linalg.solve(X1.T @ X1 + a * np.eye(2), X1.T @ yd); wb_s = np.linalg.solve(X1.T @ X1 + a * np.eye(2), X1.T @ (60 * yd))
print("seconds / minutes, penalised fit:", wb_s / wb_m)
print("prediction at 6 km, minutes: unpenalised", round(float(Ridge(alpha=a).fit(xd[:, None], yd).predict([[6.]])[0]), 3), " penalised", round(float(wb_m @ [1, 6]), 3))

print("\n== Q3  every sample twice")
for a1, a2 in ((1, 1), (1, 2), (0.5, 1)):
    w1 = Ridge(alpha=a1).fit(X, y); w2 = Ridge(alpha=a2).fit(np.r_[X, X], np.r_[y, y])
    print(f"original alpha = {a1}: w = {w1.coef_}, b = {w1.intercept_:.4f}   |   doubled data, alpha = {a2}: w = {w2.coef_}, b = {w2.intercept_:.4f}")

print("\n== Q1 (continued)  the collinearity experiment: weights and predictions on fresh data (true weights 1, 1; sigma = 1)")
rng = np.random.default_rng(1)
def batch(rho):
    D = []
    for _ in range(300):
        z = rng.normal(size=(50, 2)); a_ = z[:, 0]; b_ = rho * z[:, 0] + np.sqrt(1 - rho ** 2) * z[:, 1]
        D.append((np.c_[a_, b_], a_ + b_ + rng.normal(0, 1, 50)))
    return D
batch(0.0); D = batch(0.999)
def fresh(rho, m=4000):
    z = rng.normal(size=(m, 2)); a_ = z[:, 0]; b_ = rho * z[:, 0] + np.sqrt(1 - rho ** 2) * z[:, 1]; return np.c_[a_, b_], a_ + b_ + rng.normal(0, 1, m)
Fa, Fb = fresh(0.999), fresh(0.0)
print("alpha   rms distance of (w1, w2) from (1, 1)   test MSE, same correlation   test MSE, correlation broken")
for a in (0, 1):
    ms = [(LinearRegression() if a == 0 else Ridge(alpha=a)).fit(Xx, yy) for Xx, yy in D]
    W = np.array([m.coef_ for m in ms])
    ea = np.mean([np.mean((m.predict(Fa[0]) - Fa[1]) ** 2) for m in ms]); eb = np.mean([np.mean((m.predict(Fb[0]) - Fb[1]) ** 2) for m in ms])
    print(f"{a:4d} {np.sqrt(((W - 1) ** 2).sum(1).mean()):26.3f} {ea:30.3f} {eb:30.3f}")

print("\n== Q4  the collinearity experiment (same generator as Figures 9.9-2 and 10.10-1)")
lam = np.array([np.linalg.eigvalsh((Xx - Xx.mean(0)).T @ (Xx - Xx.mean(0))) for Xx, _ in D])
print("average eigenvalues of X'X (weak, strong):", lam.mean(0).round(3), "  50(1 - rho), 50(1 + rho) =", 50 * 0.001, 50 * 1.999)
for a in (0, 1):
    W = np.array([(LinearRegression() if a == 0 else Ridge(alpha=a)).fit(Xx, yy).coef_ for Xx, yy in D])
    print(f"alpha = {a}: sd of w1 = {W[:, 0].std():.3f}  sd of w1 + w2 = {W.sum(1).std():.3f}  sd of w1 - w2 = {(W[:, 0] - W[:, 1]).std():.3f}")
l_w, l_s = 50 * 0.001, 50 * 1.999
for a in (0, 1):
    vs = 1 / l_s if a == 0 else l_s / (l_s + a) ** 2; vw = 1 / l_w if a == 0 else l_w / (l_w + a) ** 2     # variance of c_j, Eq. 10.4-7, sigma = 1
    print(f"theory alpha = {a}: sd(w1 + w2) = {np.sqrt(2 * vs):.3f}  sd(w1 - w2) = {np.sqrt(2 * vw):.3f}  sd(w1) = {np.sqrt((2 * vs + 2 * vw) / 4):.3f}")

print("\n== Q5  stability of the diabetes weights (200 bootstrap samples of the training part)")
d = load_diabetes(); Xd, t = d.data, d.target
Xtr, Xte, ytr, yte = train_test_split(Xd, t, test_size=0.25, random_state=0)
print("correlation of s1 and s2 in the training part:", np.corrcoef(Xtr[:, 4], Xtr[:, 5])[0, 1].round(3))
rng = np.random.default_rng(0); Wo, Wr = [], []
for _ in range(200):
    i = rng.integers(0, len(ytr), len(ytr))
    Wo.append(LinearRegression().fit(Xtr[i], ytr[i]).coef_); Wr.append(Ridge(alpha=0.0501).fit(Xtr[i], ytr[i]).coef_)
Wo, Wr = np.array(Wo), np.array(Wr)
print("feature   sd least squares   sd ridge(0.0501)")
for j in (2, 4, 5, 6):
    print(f"  {d.feature_names[j]:4s} {Wo[:, j].std():13.1f} {Wr[:, j].std():14.1f}")
print("sign of s1 weight negative in OLS bootstrap:", (Wo[:, 4] < 0).mean().round(2), " s2 weight positive:", (Wo[:, 5] > 0).mean().round(2))
