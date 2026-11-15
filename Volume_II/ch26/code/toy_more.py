# Further hand calculations of Chapter 26, checked by script.
import numpy as np
from fractions import Fraction as Fr
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# --- Example 26.5-1: exact fractions of the three rounds
x = np.array([[1], [2], [3], [4], [5], [6.]])
y = np.array([-1, -1, -1, 1, -1, 1])
ada = AdaBoostClassifier(n_estimators=3, random_state=0).fit(x, y)
w = np.full(6, 1 / 6)
Z, F = [], np.zeros(6)
for m, (h, a) in enumerate(zip(ada.estimators_, ada.estimator_weights_)):
    p = h.predict(x)
    wrong = p != y
    err = w[wrong].sum()
    print(f"round {m+1}: w used", [str(Fr(v).limit_denominator(60)) for v in w],
          " err", Fr(err).limit_denominator(60))
    Z.append(2 * np.sqrt(err * (1 - err)))
    F += a / 2 * p                                   # textbook scale
    w = w * np.exp(a * wrong); w /= w.sum()
    if m == 0:
        w_after1 = w.copy()
print("Z_m", np.round(Z, 4), " product", round(np.prod(Z), 4))
print("mean exp(-y F/2)", round(np.mean(np.exp(-y * F)), 4),
      " training error", np.mean(np.sign(F) != y))
print("stump 1 under the round-2 weights, error:",
      round(np.sum(w_after1 * (ada.estimators_[0].predict(x) != y)), 4))

# --- Example 26.5-2: one round of AdaBoost.R2
xr = np.arange(1, 7.).reshape(-1, 1)
yr = np.array([2, 3, 4, 8, 9, 13.])
p = np.array([3, 3, 3, 10, 10, 10.])                 # the stump
e = np.abs(yr - p)
for loss in ("linear", "square", "exponential"):
    L = e / e.max()
    L = {"linear": L, "square": L ** 2, "exponential": 1 - np.exp(-L)}[loss]
    eps = L.mean()                                   # all w = 1/6
    beta = eps / (1 - eps)
    wn = beta ** (1 - L)
    wn /= wn.sum()
    print(f"{loss:12s} L", np.round(L, 4), "eps", round(eps, 4),
          "beta", round(beta, 4), "alpha", round(np.log(1 / beta), 4))
    print("             new w", np.round(wn, 4))

# --- the recursion reproduces scikit-learn's errors and weights
reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=1),
                        n_estimators=5, random_state=1).fit(xr, yr)
w = np.full(6, 1 / 6)
for m, h in enumerate(reg.estimators_):
    e = np.abs(yr - h.predict(xr)); L = e / e.max()
    eps = (w * L).sum(); beta = eps / (1 - eps)
    print(f"stump {m+1}: eps {eps:.4f} vs {reg.estimator_errors_[m]:.4f}"
          f"  alpha {np.log(1/beta):.4f} vs {reg.estimator_weights_[m]:.4f}")
    w = w * beta ** (1 - L); w /= w.sum()

# --- the weighted median at x = 4
P = np.array([h.predict([[4.]])[0] for h in reg.estimators_])
a = reg.estimator_weights_
o = np.argsort(P)
cdf = np.cumsum(a[o])
print("sorted predictions", P[o], " weights", np.round(a[o], 3))
print("cumulative", np.round(cdf, 3), " half of total", round(a.sum() / 2, 3))
print("weighted mean", round(float(a @ P / a.sum()), 4))
print("median", P[o][np.argmax(cdf >= a.sum() / 2)],
      " predict", reg.predict([[4.]])[0])

# --- a stump with err = 1/2 in three classes still gets alpha > 0
K = 3
print("K = 3, err = 1/2: alpha =", round(np.log(0.5 / 0.5) + np.log(K - 1), 4),
      "; the limit 1 - 1/K =", round(1 - 1 / K, 4))

# --- perfect fit stops the loop
xs = np.array([[0.], [1.], [2.], [3.]]); ys = np.array([0, 1, 0, 1])
st = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                        n_estimators=10, random_state=0).fit(xs, ys)
print("perfect fit: rounds", len(st.estimators_), " weights",
      st.estimator_weights_[:3], " errors", st.estimator_errors_[:3])
