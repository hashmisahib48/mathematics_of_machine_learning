# The three worked examples of Chapter 46, checked in NumPy.
import numpy as np
from scipy.stats import chi2
from sklearn.covariance import (LedoitWolf, OAS, ShrunkCovariance,
                                MinCovDet)
np.set_printoptions(precision=4, suppress=True)
X = np.array([[5, 5], [6, 9], [2, 3], [4, 1], [8, 12.]])
n, p = X.shape
xbar = X.mean(0)
D = X - xbar
S = D.T @ D / n                              # Eq. 46.3-3
print("mean", xbar, " S", S.ravel(), " S with n-1", (S*n/(n-1)).ravel())
Th = np.linalg.inv(S)
print("det S", np.linalg.det(S).round(4), " precision", Th.ravel())
print("d^2 of the five days", np.einsum("ij,jk,ik->i", D, Th, D).round(3))
for q in ([9, 12], [2, 8]):                 # §46.2 (A and B)
    d = np.array(q) - xbar
    print("point", q, "d^2", (d @ Th @ d).round(4), "Euclid^2", d @ d)
print("rho", (S[0, 1]/np.sqrt(S[0, 0]*S[1, 1])).round(4), " partial",
      (-Th[0, 1]/np.sqrt(Th[0, 0]*Th[1, 1])).round(4))
mu = np.trace(S) / p                         # Example 46.7-1
lam = np.linalg.eigvalsh(S)
print("\nmu", mu, " eig S", lam, " cond", (lam[1]/lam[0]).round(2))
Sa = 0.5 * S + 0.5 * mu * np.eye(p)
la = np.linalg.eigvalsh(Sa)
print("alpha 0.5:", Sa.ravel(), "eig", la, "cond", (la[1]/la[0]).round(3))
print("  ShrunkCovariance", ShrunkCovariance(shrinkage=.5).fit(X).covariance_.ravel())
sq = [float(((np.outer(d, d) - S) ** 2).sum()) for d in D]
beta = sum(sq) / (n**2 * p)
delta = ((S - mu * np.eye(p)) ** 2).sum() / p
print("sum of squares", sq, sum(sq), " beta", beta, " delta", delta)
print("alpha_LW", round(min(beta, delta)/delta, 4), " sklearn",
      LedoitWolf().fit(X).shrinkage_.round(4))
Sl = LedoitWolf().fit(X).covariance_
print("LW covariance", Sl.ravel(), "eig", np.linalg.eigvalsh(Sl))
al = (S ** 2).mean()
rho = (al + mu**2) / ((n + 1) * (al - mu**2 / p))
print("OAS: mean(S^2)", al, "mu^2", mu**2, "rho", rho.round(4),
      " sklearn", OAS().fit(X).shrinkage_.round(4))
h = int(np.ceil((n + p + 1) / 2))            # Example 46.7-2
print("\nh =", h)
for out in range(n):
    idx = [i for i in range(n) if i != out]
    print("leave out", out + 1, X[out], "det",
          np.linalg.det(np.cov(X[idx].T, bias=True)).round(4))
H = [0, 2, 3, 4]                             # start: leave out day 2
m1, S1 = X[H].mean(0), np.cov(X[H].T, bias=True)
d2 = np.einsum("ij,jk,ik->i", X - m1, np.linalg.inv(S1), X - m1)
print("start", [i+1 for i in H], "mean", m1, "S", S1.ravel(),
      "det", np.linalg.det(S1).round(4))
print("  d^2", d2.round(3), "sum over H", d2[H].sum().round(3), "= h p =", h*p)
H2 = np.sort(np.argsort(d2)[:h])
m2, S2 = X[H2].mean(0), np.cov(X[H2].T, bias=True)
print("C-step ->", [int(i)+1 for i in H2], "mean", m2, "S", S2.ravel(), "det", np.linalg.det(S2).round(4))
d3 = np.einsum("ij,jk,ik->i", X - m2, np.linalg.inv(S2), X - m2)
print("  d^2", d3.round(3), " same support again:", [int(i)+1 for i in np.sort(np.argsort(d3)[:h])])
mcd = MinCovDet(random_state=0).fit(X)
c = (h/n) / chi2.cdf(chi2.ppf(h/n, p), p + 2)
print("MinCovDet raw support", np.where(mcd.raw_support_)[0]+1, "raw cov",
      mcd.raw_covariance_.ravel(), " c =", round(c, 4))
print("  corrected d^2", (d3 / c).round(3), " cut", chi2(p).isf(0.025).round(3))
print("  final support", np.where(mcd.support_)[0]+1, "location", mcd.location_,
      "covariance", mcd.covariance_.ravel(), " c(.975) =",
      round(0.975/chi2.cdf(chi2.ppf(0.975, p), p+2), 4))
print("\nC-steps from each of the five starts (leave one day out):")
for out in range(n):
    H = [i for i in range(n) if i != out]
    while True:
        m_, S_ = X[H].mean(0), np.cov(X[H].T, bias=True)
        d_ = np.einsum("ij,jk,ik->i", X - m_, np.linalg.inv(S_), X - m_)
        Hn = sorted(np.argsort(d_)[:h].tolist())
        if Hn == sorted(H):
            break
        H = Hn
    print("  leave out", out + 1, "-> fixed point", [i + 1 for i in H],
          "det", np.linalg.det(S_).round(4))
