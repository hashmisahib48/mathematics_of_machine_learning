# Problem 50C.1: ridge with all indicators or with category a dropped
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

cat = np.array(list("aabccc")); y = np.array([4, 6, 10, 3, 5, 7.])
D = (cat[:, None] == np.unique(cat)).astype(float)
n, ybar_k = D.sum(0), D.T @ y / D.sum(0)
print("n_k", n, " category means", ybar_k, " weighted mean of the means", (n * ybar_k).sum() / n.sum(), " plain mean of the means %.4f" % ybar_k.mean())

def closed_all(alpha):                  # intercept free, all three weights penalised
    wt = n * alpha / (n + alpha); b0 = (wt * ybar_k).sum() / wt.sum()
    return b0, n * (ybar_k - b0) / (n + alpha)
def closed_drop(alpha):                 # category a is the reference: its weight is not in the model
    wt = np.array([n[0], n[1] * alpha / (n[1] + alpha), n[2] * alpha / (n[2] + alpha)]); b0 = (wt * ybar_k).sum() / wt.sum()
    return b0, n[1:] * (ybar_k[1:] - b0) / (n[1:] + alpha)
for alpha in (1.0, 1e6):
    ra, rd = Ridge(alpha).fit(D, y), Ridge(alpha).fit(D[:, 1:], y)
    b0, w = closed_all(alpha); c0, v = closed_drop(alpha)
    print("\nalpha =", alpha)
    print("  all three : intercept %.4f  coefficients %s   closed form %.4f %s" % (ra.intercept_, ra.coef_.round(4), b0, w.round(4)))
    print("  a dropped : intercept %.4f  coefficients %s   closed form %.4f %s" % (rd.intercept_, rd.coef_.round(4), c0, v.round(4)))
    print("  fitted value of a, b, c (all three):", (ra.intercept_ + ra.coef_).round(4), "  (a dropped):", np.array([rd.intercept_, *(rd.intercept_ + rd.coef_)]).round(4))
for nm, F in (("all three", D), ("a dropped", D[:, 1:])):
    lr = LinearRegression().fit(F, y); print("no penalty,", nm, ": fitted value of a, b, c", np.unique(lr.predict(F).round(6)), " coefficients", lr.coef_.round(4), " intercept %.4f" % lr.intercept_)
