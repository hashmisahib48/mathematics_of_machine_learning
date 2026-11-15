# Numbers quoted in the model answers to the Questions for discussion of Chapter 49
import numpy as np
from scipy.stats import skew
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, root_mean_squared_error as rmse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (StandardScaler, PowerTransformer, QuantileTransformer, Normalizer)
from sklearn.tree import DecisionTreeClassifier

print("== Question 1: who cares about units")
Xd, yd = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, random_state=0)
big = np.ones(Xd.shape[1]); big[2] = 1000.0           # bmi column in other units
for name, mk in (("OLS", LinearRegression), ("Ridge(alpha=1)", lambda: Ridge(1.0))):
    a = mk().fit(Xtr, ytr); b = mk().fit(Xtr * big, ytr)
    pa, pb = a.predict(Xte), b.predict(Xte * big)
    print(f"{name:15s} bmi coefficient {a.coef_[2]:10.3f} -> {b.coef_[2]:10.5f}   max |prediction change| {np.abs(pa - pb).max():.2e}   test RMSE {rmse(yte, pa):.3f} -> {rmse(yte, pb):.3f}")
Xb, yb = load_breast_cancer(return_X_y=True)
Btr, Bte, btr, bte = train_test_split(Xb, yb, test_size=0.25, random_state=0, stratify=yb)
sc = StandardScaler().fit(Btr)
for name, mk in (("DecisionTree", lambda: DecisionTreeClassifier(random_state=0)), ("RandomForest", lambda: RandomForestClassifier(100, random_state=0)),
                 ("Logistic C=1", lambda: LogisticRegression(max_iter=5000))):
    m1 = mk().fit(Btr, btr); m2 = mk().fit(sc.transform(Btr), btr)
    p1, p2 = m1.predict(Bte), m2.predict(sc.transform(Bte))
    print(f"{name:15s} test accuracy raw {m1.score(Bte, bte):.4f}  scaled {m2.score(sc.transform(Bte), bte):.4f}  predictions identical: {bool((p1 == p2).all())}")

print("\n== Question 2: counts with zeros")
rng = np.random.RandomState(0)
cnt = rng.poisson(rng.lognormal(0.8, 1.2, 1000)).astype(float).reshape(-1, 1)
print("share of zeros %.3f, max %d, skewness %.3f" % ((cnt == 0).mean(), cnt.max(), skew(cnt)[0]))
try:
    PowerTransformer(method="box-cox").fit(cnt)
except ValueError as e:
    print("box-cox:", str(e)[:70])
pt = PowerTransformer().fit(cnt)
print("log1p skew %.3f;  Yeo-Johnson lambda %.3f, skew %.3f;  Box-Cox of (x+1) lambda %.3f" % (skew(np.log1p(cnt))[0], pt.lambdas_[0], skew(pt.transform(cnt))[0],
      PowerTransformer(method="box-cox").fit(cnt + 1).lambdas_[0]))
for x0 in (0, 1, 3, 9):
    print("  x =", x0, " log1p %.3f   Yeo-Johnson (before standardising) %.3f" % (np.log1p(x0), ((x0 + 1) ** pt.lambdas_[0] - 1) / pt.lambdas_[0]))
cnt2 = np.random.RandomState(1).poisson(np.random.RandomState(1).lognormal(0.8, 1.2, 1000)).astype(float).reshape(-1, 1)
print("lambda refitted on a second sample of 1000: %.3f" % PowerTransformer().fit(cnt2).lambdas_[0])

print("\n== Question 3: the clamp of the quantile map")
rng = np.random.RandomState(0); x = rng.uniform(0, 10, (300, 1)); y = 3 * x.ravel() + 1 + rng.normal(0, 0.5, 300)
new = np.array([[10.0], [20.0], [1000.0]])
lin = LinearRegression().fit(x, y)
qt = QuantileTransformer(n_quantiles=100).fit(x)
linq = LinearRegression().fit(qt.transform(x), y)
print("true line at 10, 20, 1000:", (3 * new.ravel() + 1).round(1))
print("OLS on x          :", lin.predict(new).round(1))
print("OLS on quantile(x):", linq.predict(qt.transform(new)).round(1), " transformed values", qt.transform(new).ravel())
qn = QuantileTransformer(n_quantiles=100, output_distribution="normal").fit(x)
print("normal output at 10, 20, 1000:", qn.transform(new).ravel().round(3))

print("\n== Question 4: Normalizer")
data = load_breast_cancer(); X, yc = data.data, data.target
norm = np.linalg.norm(X, axis=1)
print("row norm: mean malignant %.1f, mean benign %.1f; AUC of the norm alone %.3f" % (norm[yc == 0].mean(), norm[yc == 1].mean(), roc_auc_score(yc == 0, norm)))
Xn = Normalizer().fit_transform(X)
print("after Normalizer every row has norm", np.unique(np.linalg.norm(Xn, axis=1).round(6)))
print("share of squared norm carried by the two area columns: %.4f" % ((X[:, [3, 23]] ** 2).sum() / (X ** 2).sum()))
d1, d2 = np.array([[2.0, 1, 0]]), np.array([[20.0, 10, 0]])
print("two documents", d1, d2, "-> Normalizer:", Normalizer().transform(d1).round(3), Normalizer().transform(d2).round(3))
