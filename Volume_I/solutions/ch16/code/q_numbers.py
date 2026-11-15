# Numbers quoted in the model answers to Questions 1 - 4 of Chapter 16
import numpy as np, warnings
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
np.set_printoptions(precision=4, suppress=True)
sig = lambda z: 1 / (1 + np.exp(-z))
x = np.arange(1., 9); y = np.array([0, 0, 0, 1, 0, 1, 1, 1.])
m = LogisticRegression(C=np.inf, tol=1e-10, max_iter=1000).fit(x[:, None], y)
b, w = float(m.intercept_[0]), float(m.coef_[0, 0])

print("== Q1  why not a line")
line = np.polyfit(x, y, 1); print("least-squares line:", line, " at 1, 8, 10 hours:", np.polyval(line, [1, 8, 10]))
print("logistic fit b, w:", b, w, " p at 0, 1, 8, 10, 20 h:", sig(b + w * np.array([0, 1, 8, 10, 20.])))
print("odds multiplier per hour e^w:", np.exp(w), " z at 4, 5, 6 hours:", b + w * np.array([4, 5, 6.]), " ln 2 =", np.log(2))
print("odds at 4, 5, 6 h:", np.exp(b + w * np.array([4, 5, 6.])), " ratios:", np.exp(w * 1))
print("log-odds of p = 0.001, 0.5, 0.999:", np.log(np.array([0.001, 0.5, 0.999]) / (1 - np.array([0.001, 0.5, 0.999]))))

print("\n== Q2  the contribution (p - y) x of each student")
p = sig(b + w * x); err = p - y
print("p:", p); print("p - y:", err); print("x (p - y):", x * err, " sum:", (x * err).sum(), " sum of p - y:", err.sum())
print("sum of |x (p - y)| of the six correct students:", np.abs(x * err)[[0, 1, 2, 5, 6, 7]].sum(), "  the two wrong ones:", np.abs(x * err)[[3, 4]].sum())
p1 = sig(b + w * 1); print("hypothetical: a student with 1 hour who PASSED: p =", p1, " p - y =", p1 - 1, " contribution to dJ/dw:", 1 * (p1 - 1),
      " squared error through the sigmoid, (p - y) p (1 - p) x:", (p1 - 1) * p1 * (1 - p1))
print("the same student who FAILED (correct, confident): p - y =", p1, " contribution", p1)

print("\n== Q3  lambda = 1/(C n), and C n is what counts")
for n in (426, 4260): print(f"n = {n}: lambda = 1/(C n) = {1 / (1 * n):.6f}")
print("C for the second analyst to keep lambda:", 426 / 4260)
X, t = load_breast_cancer(return_X_y=True)
Xtr, Xte, ttr, tte = train_test_split(X, t, test_size=0.25, random_state=0, stratify=t)
sc = StandardScaler().fit(Xtr); A, At = sc.transform(Xtr), sc.transform(Xte)
fit = lambda C, A_, t_: LogisticRegression(C=C, tol=1e-10, max_iter=10000).fit(A_, t_)
a = fit(1.0, A, ttr); A10 = np.vstack([A] * 10); t10 = np.tile(ttr, 10)
b1 = fit(1.0, A10, t10); b2 = fit(0.1, A10, t10)
print("n =", len(ttr), " |w| for C = 1:", np.linalg.norm(a.coef_), " 10 copies (n = %d) with C = 1: %.4f   with C = 0.1: %.4f" % (len(t10), np.linalg.norm(b1.coef_), np.linalg.norm(b2.coef_)))
print("largest difference of coefficients, C = 0.1 on 10 copies against C = 1 on one copy:", np.abs(b2.coef_ - a.coef_).max(), " C = 1 on 10 copies:", np.abs(b1.coef_ - a.coef_).max())

print("\n== Q4  C = infinity against C = 1 on the breast-cancer data (the split of section 16.12)")
for C in (np.inf, 1.0):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always"); mm = fit(C, A, ttr)
    pt = mm.predict_proba(At)
    print(f"C = {C:6.3g}: |w| = {np.linalg.norm(mm.coef_):10.2f}  train acc {mm.score(A, ttr):.4f}  test acc {mm.score(At, tte):.4f}  "
          f"test log loss {log_loss(tte, pt):.4f}  largest |z| on test {np.abs(mm.decision_function(At)).max():9.1f}  n_iter {mm.n_iter_[0]}  warnings {len(rec)}")
mm = fit(np.inf, A, ttr); pt = mm.predict_proba(At); wrong = mm.predict(At) != tte
print("with C = inf, the confidence on the test errors: p of the predicted class", np.where(wrong, pt.max(1), np.nan)[wrong])
