# Numbers quoted in the model answers to the questions of Chapter 47.
import numpy as np
from scipy.stats import norm
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KernelDensity, NearestNeighbors


def waits(n, seed):                      # the book's invented waiting times
    rng = np.random.default_rng(seed)
    short = rng.random(n) < 0.35
    return np.where(short, rng.normal(54, 5, n), rng.normal(80, 6, n))[:, None]


def truth(x, m1=54., m2=80.):
    return 0.35 * norm.pdf(x, m1, 5) + 0.65 * norm.pdf(x, m2, 6)


X, Xt = waits(200, 1), waits(5000, 2)
g = np.linspace(25, 115, 9001); dg = g[1] - g[0]

print("== Question 1: histogram against kernel estimate (200 waits, true density known)")
def hist_curve(w, edge):
    e = np.arange(edge - 100, 200, w)
    cnt, _ = np.histogram(X[:, 0], bins=e)
    dens = cnt / (len(X) * w)
    idx = np.clip(np.searchsorted(e, g, side="right") - 1, 0, len(dens) - 1)
    return dens[idx], dens
ise = lambda f: float(((f - truth(g)) ** 2).sum() * dg)
kde = np.exp(KernelDensity(bandwidth=2.5).fit(X).score_samples(g[:, None]))
print(f"KDE h = 2.5: integrated squared error {ise(kde):.2e}")
for w in (4, 6, 8):
    r = []
    for edge in np.arange(40, 40 + w, w / 4):
        f, dens = hist_curve(w, edge)
        r.append((ise(f), (np.abs(np.diff(dens)) > 0).sum(), dens.max()))
    print(f"histogram width {w}: ISE for 4 edge positions " + " ".join(f"{a:.2e}" for a, _, _ in r) +
          f"; tallest bar {min(c for _,_,c in r):.4f} to {max(c for _,_,c in r):.4f}")
f, dens = hist_curve(6, 40)
print("histogram w = 6, edge 40: jumps in the curve:", int((np.abs(np.diff(dens)) > 0).sum()), "occupied bins", int((dens > 0).sum()))
print("KDE: largest |slope| per minute", round(float(np.abs(np.diff(kde)).max() / dg), 4),
      " histogram largest jump", round(float(np.abs(np.diff(dens)).max()), 4))
for h in (0.8, 2.5, 10.0):
    k = np.exp(KernelDensity(bandwidth=h).fit(X).score_samples(g[:, None]))
    print(f"KDE h = {h:4.1f}: ISE {ise(k):.2e}, local maxima {int(((k[1:-1] > k[:-2]) & (k[1:-1] > k[2:])).sum())}")

print("\n== Question 2: training and held-out mean ln f against h")
print("   h   training  held-out(5000)   -ln(n h sqrt(2 pi))")
for h in (5.0, 2.5, 1.0, 0.5, 0.05, 0.005):
    k = KernelDensity(bandwidth=h).fit(X)
    tr = k.score_samples(X).mean()
    te = k.score_samples(Xt).mean()
    print(f"{h:6.3f} {tr:9.3f} {te:12.3f} {-np.log(200 * h * np.sqrt(2 * np.pi)):12.3f}")
k = KernelDensity(bandwidth=0.05).fit(X)
d = np.abs(Xt - X.T).min(1)
print("h = 0.05: the median distance of a test wait to the nearest training wait", round(float(np.median(d)), 3),
      " (in units of h:", round(float(np.median(d) / 0.05), 1), ")")
print("true density, test mean ln f:", round(float(np.log(truth(Xt[:, 0])).mean()), 4))

print("\n== Question 3: Scott's rule for one hump and for two humps")
n = 200
print("   gap  sd(pooled)  R(f'')      h*(Eq.47.4-3)  Scott h=sd*n^-1/5   ratio")
def rfpp(f):
    d2 = np.gradient(np.gradient(f, dg), dg)
    return float((d2 ** 2).sum() * dg)
RK, s2K = 1 / (2 * np.sqrt(np.pi)), 1.0
for gap in (0, 6, 12, 18, 26, 40, 60):
    m1, m2 = 67.0 - 0.65 * gap, 67.0 + 0.35 * gap           # keeps the overall mean at 67
    f = truth(g, m1, m2)
    mean = (g * f).sum() * dg; sd = np.sqrt(((g - mean) ** 2 * f).sum() * dg)
    R = rfpp(f); hs = (RK / (s2K ** 2 * R * n)) ** 0.2
    print(f"{gap:5d} {sd:9.2f} {R:12.3e} {hs:10.3f} {sd * n ** -0.2:14.3f} {sd * n ** -0.2 / hs:10.2f}")
f = truth(g); mean = (g * f).sum() * dg; sd = np.sqrt(((g - mean) ** 2 * f).sum() * dg)
print("book mixture: sd", round(sd, 3), " h* =", round((RK / (rfpp(f) * n)) ** 0.2, 3), " Scott", round(sd * n ** -0.2, 3), " 1.0592 s n^-1/5", round(1.0592 * sd * n ** -0.2, 3))
sig = 6.0; fn = norm.pdf(g, 67, sig)
print("one normal hump (sd 6): h* =", round((RK / (rfpp(fn) * n)) ** 0.2, 3), " 1.0592 s n^-1/5 =", round(1.0592 * sig * n ** -0.2, 3), " Scott", round(sig * n ** -0.2, 3))
print("for two separated humps R(f'') = sum w^2 R(f_k''):", end=" ")
R1 = 3 / (8 * np.sqrt(np.pi) * 5 ** 5); R2 = 3 / (8 * np.sqrt(np.pi) * 6 ** 5)
print(f"{0.35**2 * R1 + 0.65**2 * R2:.3e}  against numerical {rfpp(truth(g)):.3e}")

print("\n== Question 4: draws from KernelDensity against the confidential digits")
Xd, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xd, y, test_size=0.25, random_state=0, stratify=y)
pca = PCA(20, random_state=0).fit(Xtr)
Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
nn = NearestNeighbors(n_neighbors=2).fit(Ztr)
dte = nn.kneighbors(Zte, 1)[0][:, 0]
# nearest other training digit, for the training digits themselves
dtr = nn.kneighbors(Ztr, 2)[0][:, 1]
print(f"nearest training digit: held-out digits median {np.median(dte):.2f}, training digits (nearest other one) median {np.median(dtr):.2f}")
for h in (3.0, 1.0, 0.5):
    kde = KernelDensity(bandwidth=h).fit(Ztr)
    Zn = kde.sample(1000, random_state=0)
    dn = nn.kneighbors(Zn, 1)[0][:, 0]
    print(f"KDE h = {h}: 1000 draws, median distance to the nearest training digit {np.median(dn):.2f}; share closer than {np.median(dte):.2f}: {np.mean(dn < np.median(dte)):.2f};"
          f" expected displacement h*sqrt(20) = {h * np.sqrt(20):.2f}")
gm = GaussianMixture(10, reg_covar=1e-3, n_init=3, random_state=0).fit(Ztr)
Zg = gm.sample(1000)[0]
dg_ = nn.kneighbors(Zg, 1)[0][:, 0]
print(f"mixture of 10: median distance {np.median(dg_):.2f}, share closer than the held-out median {np.mean(dg_ < np.median(dte)):.2f}")
kde = KernelDensity(bandwidth=3.0).fit(Ztr)
rng = np.random.default_rng(0); parent = rng.integers(0, len(Ztr), 1000)
Zn = Ztr[parent] + 3.0 * rng.normal(size=(1000, 20))
j = nn.kneighbors(Zn, 1)[1][:, 0]
print("h = 3: share of draws whose nearest training digit is the one they were built on:", round(float(np.mean(j == parent)), 3))
print("same label as the parent digit:", round(float(np.mean(ytr[j] == ytr[parent])), 3))
