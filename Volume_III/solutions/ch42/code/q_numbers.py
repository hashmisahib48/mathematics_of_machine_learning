# Numbers quoted in the model answers to the questions for discussion (Q1 - Q4) of Chapter 42
import numpy as np
from sklearn.datasets import load_digits, load_wine
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import f_classif
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
np.set_printoptions(precision=4, suppress=True)

F = np.array([[27, 4], [29, 6], [30, 6], [31, 8], [33, 6]], float); Fc = F - F.mean(0)

print("== Q1  three lines through the fish")
S = Fc.T @ Fc / 4; print("S =", S.tolist())
b_yx = S[0, 1] / S[0, 0]; b_xy = S[0, 1] / S[1, 1]
print("regression of weight on length: slope %.4f (angle %.2f deg); minimises vertical distances" % (b_yx, np.degrees(np.arctan(b_yx))))
print("regression of length on weight: dy/dx = 1/slope = %.4f (angle %.2f deg); minimises horizontal distances" % (1 / b_xy, np.degrees(np.arctan(1 / b_xy))))
w = PCA(1).fit(F).components_[0]; print("first principal axis: slope %.4f (angle %.2f deg); minimises perpendicular distances" % (w[1] / w[0], np.degrees(np.arctan(w[1] / w[0]))))
def cost(th):                                          # variance of shadows and squared distances for the direction th
    u = np.array([np.cos(th), np.sin(th)]); z = Fc @ u; return z.var(ddof=1), ((Fc - np.outer(z, u)) ** 2).sum()
for deg in (0, 21.8, 26.57, 45, 90):
    v, d = cost(np.radians(deg)); print("direction %5.2f deg: variance of shadows %.4f, squared distances %.4f, sum of squares |xc|^2 = %.4f" % (deg, v, d, (Fc ** 2).sum()))
print("variance*(n-1) + squared distances = %.4f for every direction (Pythagoras)" % ((Fc ** 2).sum()))
Fm = F.copy(); Fm[:, 0] *= 10                            # length in millimetres
print("length in mm: first PC explains %.4f; direction (%.4f, %.4f)" % (PCA(1).fit(Fm).explained_variance_ratio_[0], *np.abs(PCA(1).fit(Fm).components_[0])))
print("regression slope with length in mm: %.4f per mm, which is %.4f per cm: unchanged in the original units" % ((Fm - Fm.mean(0))[:, 0] @ (Fm - Fm.mean(0))[:, 1] / ((Fm - Fm.mean(0))[:, 0] ** 2).sum(), 10 * (Fm - Fm.mean(0))[:, 0] @ (Fm - Fm.mean(0))[:, 1] / ((Fm - Fm.mean(0))[:, 0] ** 2).sum()))

print("\n== Q2  signs and a different first component")
X, y = load_digits(return_X_y=True); Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
a = PCA(10, svd_solver="full").fit(Xtr); Xc = Xtr - Xtr.mean(0); e, V = np.linalg.eigh(Xc.T @ Xc / (len(Xtr) - 1)); V = V[:, ::-1][:, :10]
cos = (a.components_ * V.T).sum(1); print("cosines between the PCA and eigh components 1-10:", cos.round(4).tolist())
for name, other in (("TruncatedSVD (no centring)", TruncatedSVD(10, algorithm="arpack", random_state=0).fit(Xtr).components_),
                    ("PCA on the standardised pixels", PCA(10, svd_solver="full").fit(StandardScaler().fit_transform(Xtr)).components_)):
    print("%-32s |cos| with the PCA components 1-5:" % name, np.abs((a.components_[:5] * other[:5]).sum(1)).round(4).tolist())
Xw, yw = load_wine(return_X_y=True)
r = PCA(2).fit(Xw).components_; s = PCA(2).fit(StandardScaler().fit_transform(Xw)).components_
print("wine, raw vs standardised, |cos| of PC1: %.4f, of PC2: %.4f" % tuple(np.abs((r * s).sum(1))))
def angle_spread(l1, l2, n=100, reps=200, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(reps):
        Z = rng.normal(size=(n, 2)) * np.sqrt([l1, l2]); u = PCA(1).fit(Z).components_[0]; out.append(np.degrees(np.arctan2(u[1], u[0])) % 180)
    out = np.array(out); d = np.minimum(out, 180 - out); return d.mean(), d.std(), d.max()
for l1, l2 in ((4.0, 1.0), (1.0, 0.9)):
    m_, s_, mx = angle_spread(l1, l2); print("axis-aligned cloud with variances %.1f and %.1f, n = 100, 200 samples: angle of PC1 to the x axis mean %.1f, sd %.1f, max %.1f degrees" % (l1, l2, m_, s_, mx))
ts = TruncatedSVD(10, algorithm="arpack", random_state=0).fit(Xtr).components_
print("TruncatedSVD component j+1 against PCA component j, |cos| for j = 1..5:", np.abs((ts[1:6] * a.components_[:5]).sum(1)).round(3).tolist(), " and |cos|(TSVD 1, mean image) = %.5f" % abs(ts[0] @ Xtr.mean(0) / np.linalg.norm(Xtr.mean(0))))
ia = PCA(10, svd_solver="randomized", iterated_power=0, random_state=0).fit(Xtr)
print("randomized with no power iteration: |cos| of components 1-10 with 'full':", np.abs((ia.components_ * a.components_).sum(1)).round(3).tolist())

print("\n== Q3  why the best k is not the variance rule")
cv = lambda k: cross_val_score(make_pipeline(PCA(k), KNeighborsClassifier(5)), Xtr, ytr, cv=5)   # the five folds of the book
print("  k   mean CV   sd over folds")
for k in (13, 21, 25, 28, 41, 64):
    v = cv(k); print("%3d  %.4f   %.4f" % (k, v.mean(), v.std(ddof=1)))
pc = PCA().fit(Xtr); Z = pc.transform(Xtr); Fv, _ = f_classif(Z[:, :60], ytr); cumv = np.cumsum(pc.explained_variance_ratio_)
print("variance share and class F statistic of components:")
for j in (1, 2, 5, 10, 15, 21, 22, 25, 28, 30, 40, 50): print("  PC %2d: variance share %.4f  F %.1f" % (j, pc.explained_variance_ratio_[j - 1], Fv[j - 1]))
nz = np.arange(60); from scipy.stats import spearmanr
print("Spearman correlation between variance rank and F over the 60 components with variance: %.3f" % spearmanr(pc.explained_variance_[:60], Fv[:60])[0])
print("components beyond the 21st with F larger than the median F of the first 21: %d of 39 (median %.1f)" % ((Fv[21:60] > np.median(Fv[:21])).sum(), np.median(Fv[:21])))
rank = np.argsort(-Fv[:60]); print("the 21 components with the largest F are:", sorted((rank[:21] + 1).tolist()))
print("how many of the first 21 are among them:", sum(1 for j in rank[:21] if j < 21))

print("\n== Q4  whitening and standardising on the wine data")
Xs = StandardScaler().fit_transform(Xw); folds = StratifiedKFold(5, shuffle=True, random_state=0)
acc = lambda Z: cross_val_score(KNeighborsClassifier(5), Z, yw, cv=folds).mean()
for nm, data in (("raw", Xw), ("standardised", Xs)):
    print("%-13s" % nm, " 2 comps: plain %.4f whitened %.4f |" % (acc(PCA(2).fit_transform(data)), acc(PCA(2, whiten=True).fit_transform(data))),
          " 13 comps: plain %.4f whitened %.4f" % (acc(PCA(13).fit_transform(data)), acc(PCA(13, whiten=True).fit_transform(data))),
          " no PCA: %.4f" % acc(data))
Zr = PCA(13, whiten=True).fit_transform(Xw); Zs = PCA(13, whiten=True).fit_transform(Xs)
from scipy.spatial.distance import pdist
print("whitened with all 13 components: raw and standardised pairwise distances differ by at most %.1e (max distance %.2f)" % (np.abs(pdist(Zr) - pdist(Zs)).max(), pdist(Zr).max()))
print("cov of whitened scores = identity:", np.allclose(np.cov(Zs.T), np.eye(13)), "; correlation of the standardised features: max off-diagonal %.3f" % np.abs(np.corrcoef(Xs.T) - np.eye(13)).max())
print("PC1 of the raw data: share %.4f, PC1 of the standardised: share %.4f" % (PCA(1).fit(Xw).explained_variance_ratio_[0], PCA(1).fit(Xs).explained_variance_ratio_[0]))
