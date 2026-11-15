# Numbers quoted in the model answers to the questions for discussion (Q1 - Q6)
import time, numpy as np
from scipy.stats import norm
from sklearn.neighbors import KNeighborsClassifier, KDTree
from sklearn.linear_model import LogisticRegression
np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)

print("== Q1  cost of a prediction service")
n, p = 10_000_000, 50
print("n = %d, p = %d: multiplications per query %.2e, memory of the stored rows %.1f GB" % (n, p, n * p, n * p * 8 / 1e9))
print("requests per second on average: %.1f" % (1e6 / 86400), "  multiplications per second needed: %.2e" % (1e6 / 86400 * n * p))
n1 = 1_000_000; Xs = rng.normal(size=(n1, p)).astype(np.float32); q = rng.normal(size=p).astype(np.float32)
t0 = time.perf_counter()
for _ in range(5): d2 = ((Xs - q) ** 2).sum(1); idx = np.argpartition(d2, 5)[:5]
t = (time.perf_counter() - t0) / 5
print("one brute-force query, n = 10^6, p = 50 (float32, NumPy): %.3f s  (machine dependent); scaled to n = 10^7: %.2f s" % (t, 10 * t))
w = rng.normal(size=p).astype(np.float32); t0 = time.perf_counter()
for _ in range(200): s = float(w @ q)
print("one logistic score (50 multiplications): %.2e s" % ((time.perf_counter() - t0) / 200))
del Xs

print("\n== Q2  random labels")
for frac in (0.5, 0.3):
    Xtr = rng.normal(size=(300, 5)); ytr = (rng.uniform(size=300) < frac).astype(int)
    Xte = rng.normal(size=(5000, 5)); yte = (rng.uniform(size=5000) < frac).astype(int)
    out = []
    for k in (1, 5, 51, 300):
        m = KNeighborsClassifier(k).fit(Xtr, ytr); out.append((k, round(m.score(Xtr, ytr), 3), round(m.score(Xte, yte), 3)))
    print("share of class 1 = %.1f (training %.3f); (k, train acc, test acc):" % (frac, ytr.mean()), out,
          " theory for k=1: %.3f, majority rule: %.3f" % (frac ** 2 + (1 - frac) ** 2, max(frac, 1 - frac)))
m = KNeighborsClassifier(1).fit(Xtr, ytr); print("leave-one-out accuracy of 1-NN on the training rows (predict(None)):", round(float((m.predict(None) == ytr).mean()), 3))

print("\n== Q3  metres and millimetres")
def accs(Xa, y, name):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    A, B, ya, yb = train_test_split(Xa, y, test_size=0.5, random_state=0)
    raw = KNeighborsClassifier(5).fit(A, ya).score(B, yb); std = make_pipeline(StandardScaler(), KNeighborsClassifier(5)).fit(A, ya).score(B, yb)
    one = KNeighborsClassifier(5).fit(A[:, :1], ya).score(B[:, :1], yb)
    print("%-26s 5-NN raw %.3f   standardised %.3f   x1 alone %.3f" % (name, raw, std, one))
n = 600; u1 = rng.uniform(0, 1, n); u2 = rng.uniform(0, 1, n)       # u in metres / in units of 1 m
Xm = np.c_[u1, 1000 * u2]                                            # x1 in metres, x2 in millimetres (range 0..1000)
print("spread of x1: %.2f, spread of x2: %.0f" % (Xm[:, 0].std(), Xm[:, 1].std()))
accs(Xm, (u1 + u2 > 1).astype(int), "both informative")
accs(Xm, (u1 > 0.5).astype(int), "only x1 informative")
print("share of squared distance from x2 in the raw data: %.6f" % (Xm[:, 1].var() / (Xm[:, 0].var() + Xm[:, 1].var())))

print("\n== Q4  cosine and Euclidean distance")
a = np.array([2., 1, 0, 3]); b = 2 * a
cos = lambda u, v: 1 - u @ v / np.linalg.norm(u) / np.linalg.norm(v)
print("document a =", a, " b = 2a: Euclidean %.4f, cosine %.4f" % (np.linalg.norm(a - b), cos(a, b)))
Xd = rng.poisson(1.0, size=(400, 30)).astype(float) * rng.uniform(0.5, 5, size=(400, 1)); Xd = np.round(Xd)
Xd = Xd[np.linalg.norm(Xd, axis=1) > 0]; Qd = Xd[:50]
nb_cos = KNeighborsClassifier(6, metric="cosine").fit(Xd, np.zeros(len(Xd), int)); nb_cos = nb_cos._fit_method
from sklearn.neighbors import NearestNeighbors
ic = NearestNeighbors(n_neighbors=6, metric="cosine").fit(Xd).kneighbors(Qd, return_distance=False)
U = Xd / np.linalg.norm(Xd, axis=1, keepdims=True); tree = KDTree(U)
ie = tree.query(U[:50], k=6, return_distance=False)
print("fit method used for metric='cosine':", nb_cos, "; neighbour lists equal to Euclidean on unit-length rows:", bool((np.sort(ic, 1) == np.sort(ie, 1)).all()))
v, w_ = U[0], U[1]; print("squared Euclidean distance of unit vectors %.4f = 2 x cosine distance %.4f" % (((v - w_) ** 2).sum(), 2 * cos(v, w_)))

print("\n== Q5  Bayes error 0.05 and the one-neighbour rule")
R = 0.05; delta = -2 * norm.ppf(R)
print("Bayes error %.2f, bounds: %.4f to %.4f; centres %.3f apart" % (R, R, 2 * R * (1 - R), delta))
def sample(n, d):
    y = rng.integers(0, 2, n); X = rng.normal(size=(n, d)); X[:, 0] += delta * (y - 0.5); return X, y
Xt, yt = sample(200000, 2); eta = 1 / (1 + np.exp(-(delta * Xt[:, 0])))   # P(y=1|x) for equal priors, unit covariance
print("limit E[2 eta (1-eta)] by Monte Carlo (d-independent): %.4f" % np.mean(2 * eta * (1 - eta)))
for d in (2, 10):
    row = []
    for n in (20, 100, 300, 3000):
        errs = []
        for _ in range(10 if n < 3000 else 3):
            X, y = sample(n, d); Xq, yq = sample(4000, d); errs.append(1 - KNeighborsClassifier(1).fit(X, y).score(Xq, yq))
        row.append((n, round(float(np.mean(errs)), 3)))
    print("dimension %2d: (n, mean 1-NN error) =" % d, row)

print("\n== Q6  weights='distance' against a small k")
X = np.array([[5, 3], [6, 2], [3, 4], [4, 1], [7, 5], [8, 8], [6, 9], [9, 8]], float); y = np.array([0, 0, 0, 0, 1, 1, 1, 1]); q = [[6., 5.]]
for k, wt in ((1, "uniform"), (3, "uniform"), (3, "distance"), (5, "uniform"), (5, "distance"), (8, "distance")):
    m = KNeighborsClassifier(k, weights=wt).fit(X, y); print("k = %d %-9s class %d, share of plum %.4f" % (k, wt, m.predict(q)[0], m.predict_proba(q)[0, 1]))
Xdup = np.array([[0., 0], [0, 0], [1, 0], [3, 0], [4, 0]]); ydup = np.array([0, 1, 1, 0, 0])
for k, wt in ((1, "uniform"), (3, "uniform"), (3, "distance")):
    m = KNeighborsClassifier(k, weights=wt).fit(Xdup, ydup); print("duplicates (0,0) with labels 0 and 1; query (0.1, 0): k=%d %-9s probabilities %s" % (k, wt, m.predict_proba([[0.1, 0]])[0]),
          " query (0,0):", m.predict_proba([[0., 0]])[0])
