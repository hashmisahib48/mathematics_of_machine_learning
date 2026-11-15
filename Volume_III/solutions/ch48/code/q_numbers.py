# Numbers quoted in the model answers to the questions of Chapter 48.
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../ch48/code')
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sensors import make_sensors, MODES

def healthy(n, seed): return make_sensors(n=n, n_fault=1, seed=seed)[0][:-1]
def faults(m, seed): return make_sensors(n=10, n_fault=m, seed=seed)[0][-m:]

print("== Question 1: the score 0.5 on a healthy month")
H = healthy(300, 0); Z = StandardScaler().fit(H)
fo = IsolationForest(random_state=0).fit(Z.transform(H))
print("forest fitted on 300 healthy readings, psi =", fo.max_samples_, ": share flagged by 'auto' in the training month",
      round(float((fo.predict(Z.transform(H)) == -1).mean()), 4), "=", int((fo.predict(Z.transform(H)) == -1).sum()), "of 300")
big = healthy(30000, 5); sb = -fo.score_samples(Z.transform(big))
print("same forest on 30 000 fresh healthy readings: share with s > 0.5:", round(float((sb > 0.5).mean()), 4))
print("score quantiles of the fresh healthy readings (s): 50%", np.quantile(sb, 0.5).round(3), " 90%", np.quantile(sb, 0.9).round(3), " 99%", np.quantile(sb, 0.99).round(3), " 99.9%", np.quantile(sb, 0.999).round(3), " max", sb.max().round(3))
F = faults(500, 6); sf = -fo.score_samples(Z.transform(F)); lim = np.quantile(sb, 0.999)
print("500 faults: median s", np.median(sf).round(3), " share above 0.5:", round(float((sf > 0.5).mean()), 3), " share above the 99.9% healthy limit", round(lim, 3), ":", round(float((sf > lim).mean()), 3))
print("AUC healthy against faults:", round(roc_auc_score(np.r_[np.zeros(len(sb)), np.ones(len(sf))], np.r_[sb, sf]), 4))
rate = 0.001
print(f"fault rate {rate}: with 12 % flagged the best precision is {rate/0.12:.4f}; at 10 000 readings/day: {rate*10000:.0f} faults, {0.12*10000:.0f} flags")
# a 0.1 % limit from the 300 healthy readings of the first month is not estimable
s300 = -fo.score_samples(Z.transform(H)); print("300 healthy readings: largest s", s300.max().round(3), " 99.9% quantile estimate (=interpolated near the maximum)", np.quantile(s300, 0.999).round(3), " against the 30 000 estimate", lim.round(3))
print("  share of fresh healthy readings above the largest of the 300:", round(float((sb > s300.max()).mean()), 4))

print("\n== Question 2: where the forest and the factor agree and disagree")
def ranks(X, test_idx):
    s_if = -IsolationForest(random_state=0, n_estimators=300).fit(X).score_samples(X)
    s_lo = -LocalOutlierFactor(n_neighbors=10).fit(X).negative_outlier_factor_ if len(X) > 20 else None
    r = lambda s: int((s > s[test_idx]).sum()) + 1
    return r(s_if), r(s_lo), s_if[test_idx], s_lo[test_idx]
rng = np.random.RandomState(0)
A = np.vstack([rng.normal(size=(200, 2)), [[6, 6]]])
print("A: one Gaussian blob of 200 and a point at (6, 6): rank of the point by forest %d, by LOF %d  (scores %.3f, %.2f)" % ranks(A, 200))
dense = rng.normal(0, 0.1, (200, 2)); sparse = rng.normal(0, 1.5, (200, 2)) + [6, 0]
for xq in ([0.8, 0.0], [1.2, 0.0], [0.5, 0.0]):
    C = np.vstack([dense, sparse, [xq]])
    print("C: dense blob (sd 0.1) + sparse blob (sd 1.5) at 6 + a point at", xq, ": rank by forest %d, by LOF %d  (scores %.3f, %.2f)" % ranks(C, 400))
C = np.vstack([dense, sparse, [[0.8, 0.0]]])
s_if = -IsolationForest(random_state=0, n_estimators=300).fit(C).score_samples(C)
print("   the forest's top five of C are in the sparse blob:", bool((np.argsort(-s_if)[:5] >= 200).all()), " largest scores", np.sort(s_if)[-5:].round(3))
print("   nearest distances in C: the point to its 10th neighbour", round(float(np.sort(np.hypot(*(C - C[400]).T))[10]), 3), " typical 10th-neighbour distance in the dense blob", round(float(np.median([np.sort(np.hypot(*(dense - p).T))[10] for p in dense[:50]])), 3), " in the sparse blob", round(float(np.median([np.sort(np.hypot(*(sparse - p).T))[10] for p in sparse[:50]])), 3))
x = np.array([0, 1, 2, 3, 7, 13, 17, 21.])[:, None]
fi = IsolationForest(n_estimators=1000, max_samples=8, random_state=0).fit(x); s = -fi.score_samples(x)
lo = -LocalOutlierFactor(n_neighbors=2).fit(x).negative_outlier_factor_
print("B: the toy readings: forest s(7) = %.3f, s(21) = %.3f, rank of 21 %d, of 7 %d; LOF(7) = %.2f, LOF(21) = %.2f, rank of 7 %d, of 21 %d" % (s[4], s[7], 1 + (s > s[7]).sum(), 1 + (s > s[4]).sum(), lo[4], lo[7], 1 + (lo > lo[4]).sum(), 1 + (lo > lo[7]).sum()))

print("\n== Question 3: the mean and the standard deviation of the scaler")
Xall, y = make_sensors()
print("sd of the vibration: all 315 readings %.3f, the 300 clean ones %.3f; sd of the temperature %.3f against %.3f" % (Xall[:, 1].std(), Xall[y == 0][:, 1].std(), Xall[:, 0].std(), Xall[y == 0][:, 0].std()))
print("median and MAD-based scale of the vibration, all 315: %.3f, %.3f" % (np.median(Xall[:, 1]), 1.4826 * np.median(np.abs(Xall[:, 1] - np.median(Xall[:, 1])))))
clean = Xall[y == 0]; sc = StandardScaler().fit(clean)
oc = OneClassSVM(nu=0.05).fit(sc.transform(clean))
mu, sd, rho = MODES[0]; rr = np.random.RandomState(7)
cov = np.array([[sd[0] ** 2, rho * sd[0] * sd[1]], [rho * sd[0] * sd[1], sd[1] ** 2]])
batch = rr.multivariate_normal(mu, cov, size=50) + [8.0, 0.0]              # idle readings drifted by +8 deg C
print("a batch of 50 idle readings with the temperature drifted by +8 deg C (%.1f sd of the idle mode):" % (8 / sd[0]))
print("  scaled with the training mean and sd      : flagged", int((oc.predict(sc.transform(batch)) == -1).sum()), "of 50")
sb_ = StandardScaler().fit(batch)
print("  scaled with its own mean and sd (the leak): flagged", int((oc.predict(sb_.transform(batch)) == -1).sum()), "of 50;  the mean of the scaled temperatures", sb_.transform(batch)[:, 0].mean().round(3))
print("  scaled with the training scaler, mean z of the temperature:", sc.transform(batch)[:, 0].mean().round(2))
mix = np.vstack([clean, batch]); sm = StandardScaler().fit(mix)
print("  scaler fitted on training + batch: flagged", int((oc.predict(sm.transform(batch)) == -1).sum()), "of 50 with the old detector")
oc2 = OneClassSVM(nu=0.05).fit(sm.transform(clean))
print("  scaler fitted on training + batch and the detector refitted in that space: flagged", int((oc2.predict(sm.transform(batch)) == -1).sum()), "of 50")
FB = faults(50, 8)                                                         # a batch that really is strange
print("a batch of 50 faulty readings:")
print("  scaled with the training mean and sd      : flagged", int((oc.predict(sc.transform(FB)) == -1).sum()), "of 50")
sf_ = StandardScaler().fit(FB)
print("  scaled with its own mean and sd (the leak): flagged", int((oc.predict(sf_.transform(FB)) == -1).sum()), "of 50; mean of the scaled batch", sf_.transform(FB).mean(0).round(3), " sd", sf_.transform(FB).std(0).round(3))
print("  sd of the faulty batch %s against the training sd %s" % (FB.std(0).round(2), clean.std(0).round(2)))

print("\n== Question 4: ten million readings with 20 features")
rng = np.random.RandomState(0)
def data(n): return np.vstack([rng.normal(0, 1, (n // 2, 20)), rng.normal(0, 1, (n - n // 2, 20)) + 3.0])
Xtr = data(20000); Xq = data(1000000)
t0 = time.perf_counter(); fo = IsolationForest(n_estimators=100, random_state=0).fit(Xtr); tf = time.perf_counter() - t0
t0 = time.perf_counter(); fo.score_samples(Xq[:200000]); ts = time.perf_counter() - t0
print("forest, 100 trees, psi = %d: fit on 20 000 rows %.2f s;  scoring 200 000 rows %.2f s -> 10^7 rows about %.0f s (single process)" % (fo.max_samples_, tf, ts, ts * 50))
t0 = time.perf_counter(); IsolationForest(n_estimators=100, random_state=0).fit(data(200000)); tf2 = time.perf_counter() - t0
print("forest fit on 200 000 rows (psi still 256): %.2f s" % tf2)
t0 = time.perf_counter(); lof = LocalOutlierFactor(novelty=True).fit(Xtr); tlf = time.perf_counter() - t0
print("LOF (k = 20) fit on 20 000 rows: %.2f s; the neighbour search is quadratic, so 10^7 rows would take about %.1e s = %.0f days" % (tlf, tlf * 500 ** 2, tlf * 500 ** 2 / 86400))
t0 = time.perf_counter(); lof.score_samples(Xq[:5000]); tl = time.perf_counter() - t0
print("LOF (k = 20) fitted on 20 000 rows: scoring 5 000 rows %.2f s -> 10^7 rows about %.0f s = %.1f h" % (tl, tl * 2000, tl * 2000 / 3600))
nys = make_pipeline(Nystroem(gamma=0.02, n_components=100, random_state=0), SGDOneClassSVM(nu=0.05, tol=None, max_iter=5, random_state=0))
t0 = time.perf_counter(); nys.fit(Xq[:200000]); tn = time.perf_counter() - t0
t0 = time.perf_counter(); nys.decision_function(Xq[:200000]); tn2 = time.perf_counter() - t0
print("Nystroem(100) + SGDOneClassSVM, 5 epochs on 200 000 rows: fit %.2f s, scoring 200 000 rows %.2f s -> 10^7 rows: fit about %.0f s, scoring about %.0f s" % (tn, tn2, tn * 50, tn2 * 50))
T = json.load(open(HERE + '/../../../ch48/verify/timing.json'))
ex = [(int(n), v[0]) for n, v in T.items() if v[0] is not None]
a = np.polyfit(np.log([n for n, _ in ex[-3:]]), np.log([t for _, t in ex[-3:]]), 1)[0]
print("book timing of the exact one-class SVM: slope of ln t against ln n over the last three sizes:", round(float(a), 2))
print("   extrapolated to 10^7 rows from 5.794 s at 40 000: %.1e s = %.0f days" % (5.794 * (1e7 / 4e4) ** a, 5.794 * (1e7 / 4e4) ** a / 86400))
ap = [(int(n), v[1]) for n, v in T.items()]
b = np.polyfit(np.log([n for n, _ in ap[-3:]]), np.log([t for _, t in ap[-3:]]), 1)[0]
print("book timing of Nystroem + SGD: slope over the last three sizes:", round(float(b), 2), " (30 epochs, 100 landmarks)")
