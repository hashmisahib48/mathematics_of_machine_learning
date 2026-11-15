# Hand checks for the solutions of Chapter 45 (Problems 45A.1, 45A.2, and numbers quoted in 45B.1, 45B.2)
import numpy as np
from scipy.stats import chi2
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
np.set_printoptions(precision=4, suppress=True)

print("== 45A.1  a projection by hand (s = 3, k = 3)")
R = np.array([[1, 0, -1, 0, 1, 0], [0, 1, 0, -1, 0, -1], [-1, 0, 0, 1, 0, 1]], float)
u = np.array([3, 1, 2, 0, 1, 2.]); v = np.array([1, 1, 0, 2, 1, 0.]); d = u - v
print("entry size sqrt(s/k) =", np.sqrt(3 / 3), "  non-zeros per row:", (R != 0).sum(1).tolist())
print("d =", d.tolist(), " |d|^2 =", d @ d, " squares:", (d**2).tolist())
print("rows of R times d:", [float(r @ d) for r in R], " Rd =", R @ d, " |Rd|^2 =", (R @ d) @ (R @ d))
q = (R @ d) @ (R @ d) / (d @ d); print("q =", q)
k = 3; x4 = (d**4).sum() / (d @ d)**2
print("sum d^4 / |d|^4 = %.4f;  Var q = (2 + (s-3) * that)/k = %.4f, sd %.4f;  Gaussian sd sqrt(2/k) = %.4f" % (x4, (2 + 0 * x4) / k, np.sqrt(2 / k), np.sqrt(2 / k)))
print("z-score of q = 0.25: %.3f;  P(chi2_3/3 <= 0.25) = %.4f;  P(q <= 0.25 or q >= 1.75) = %.4f" % ((q - 1) / np.sqrt(2 / k), chi2.cdf(0.25 * k, k), chi2.cdf(0.25 * k, k) + chi2.sf(1.75 * k, k)))
rng = np.random.RandomState(0); S = rng.choice([-1., 0., 1.], p=[1 / 6, 2 / 3, 1 / 6], size=(200000, 3, 6))
qs = (S @ d)**2; qs = qs.sum(1) / (d @ d) / 1.0   # entries +-1 with probability 1/6 each: variance 1/3 = 1/k
print("simulation of 200000 matrices of Eq. 45.3-4: mean q %.3f  sd %.3f  share with q <= 0.25: %.3f" % (qs.mean(), qs.std(), (qs <= 0.25 + 1e-12).mean()))
print("if the entries were N(0, 1/3) instead: sd of q would be %.3f (same, since s = 3)" % np.sqrt(2 / 3))
print("slip: forgetting the scale, i.e. using entries +-1/sqrt(k) = %.3f gives |Rd|^2 = %.3f and q = %.3f" % (1 / np.sqrt(3), ((R / np.sqrt(3)) @ d) @ ((R / np.sqrt(3)) @ d), ((R / np.sqrt(3)) @ d) @ ((R / np.sqrt(3)) @ d) / (d @ d)))

print("\n== 45A.2  one landmark, points 0, 1, 2")
t = np.arange(3.); K = np.exp(-0.5 * (t[:, None] - t[None])**2)
C = K[:, [1]]; W = K[[1]][:, [1]]; Kh = C @ np.linalg.inv(W) @ C.T
print("K =\n", K, "\nC =", C.ravel(), " W =", W.ravel(), "\nKhat = C W^-1 C^T =\n", Kh)
E = K - Kh; print("error =\n", E, "\neigenvalues of the error:", np.linalg.eigvalsh(E), " trace", np.trace(E), " 2(1 - e^-1) =", 2 * (1 - np.exp(-1)))
print("rank of Khat:", np.linalg.matrix_rank(Kh), " exp(-1) =", np.exp(-1), " exp(-2) =", np.exp(-2), " exp(-0.5) =", np.exp(-0.5))
print("eigenvalues of Khat:", np.linalg.eigvalsh(Kh), " of K:", np.linalg.eigvalsh(K), "  relative Frobenius error %.4f" % (np.linalg.norm(E) / np.linalg.norm(K)))
print("feature map z(x) = k(x, 1) / sqrt(k(1,1)):", (C / np.sqrt(W)).ravel(), "  z(x)^2 = ", ((C / np.sqrt(W))**2).ravel())

print("\n== numbers for 45B.1: distortion of the digit distances")
X, y = load_digits(return_X_y=True); Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
sv = np.linalg.svd(Xtr - Xtr.mean(0), compute_uv=False)**2; cs = np.cumsum(sv) / sv.sum()
print("PCA of the training pixels: components for 80%%, 90%%, 95%% of the variance: %d %d %d;  first 4, 8, 16, 32: %.3f %.3f %.3f %.3f" % (
    np.searchsorted(cs, .8) + 1, np.searchsorted(cs, .9) + 1, np.searchsorted(cs, .95) + 1, cs[3], cs[7], cs[15], cs[31]))
a, b = np.random.RandomState(1).randint(0, len(Xtr), (2, 3000)); m = a != b; a, b = a[m], b[m]; d2 = ((Xtr[a] - Xtr[b])**2).sum(1)
print("sd of q over 3000 pairs, one matrix (seed 0) and the average over 10 matrices, against sqrt(2/k):")
for k in (4, 8, 16, 32):
    sds = [(((GaussianRandomProjection(k, random_state=s).fit(Xtr).transform(Xtr)[a] - GaussianRandomProjection(k, random_state=s).fit(Xtr).transform(Xtr)[b])**2).sum(1) / d2).std() for s in range(10)]
    print("  k = %2d: %.3f   theory %.3f" % (k, np.mean(sds), np.sqrt(2 / k)))
dd = Xtr[a] - Xtr[b]; r4 = (dd**4).sum(1) / (dd**2).sum(1)**2
print("digit differences: mean of sum d^4 / |d|^4 over 3000 pairs = %.4f (flat would be %.4f, spiky 1); sparse s = sqrt(64) = 8 adds (s-3) * that = %.3f to the 2 of Eq. 45.3-5" % (r4.mean(), 1 / 64, 5 * r4.mean()))
