# Numbers quoted in the model answers of Chapter 40. Run: python3 q_numbers.py
import sys, os, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../ch40/code')
warnings.simplefilter('ignore', UserWarning); warnings.simplefilter('ignore', RuntimeWarning)
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits, load_wine
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from people import people_2d, heights_1d

# ---- Question 1: soft against hard on the 400 heights
h, g = heights_1d(); H = h[:, None]
km = KMeans(2, n_init=10, random_state=0).fit(H)
gm = GaussianMixture(2, n_init=5, random_state=0).fit(H)
order = np.argsort(gm.means_.ravel()); R = gm.predict_proba(H)[:, order[1]]    # responsibility of the taller group
cs = np.sort(km.cluster_centers_.ravel())
print("Q1: 400 heights; k-means centres %.1f %.1f, midpoint %.2f" % (cs[0], cs[1], cs.mean()))
print("  mixture: weights", gm.weights_[order].round(3), "means", gm.means_.ravel()[order].round(1), "sd", np.sqrt(gm.covariances_.ravel()[order]).round(2))
xs = np.linspace(150, 190, 4001); rr = gm.predict_proba(xs[:, None])[:, order[1]]
print("  responsibility of the taller group = 0.5 at %.1f cm" % xs[np.argmin(np.abs(rr - 0.5))])
hard_k = (H.ravel() > cs.mean()); hard_g = R > 0.5
print("  people for whom the k-means cut and the mixture's 0.5 rule differ: %d of 400" % (hard_k != hard_g).sum())
print("  people with responsibility between 0.1 and 0.9: %d; between 0.2 and 0.8: %d; above 0.9: %d" % (((R > 0.1) & (R < 0.9)).sum(), ((R > 0.2) & (R < 0.8)).sum(), (R > 0.9).sum()))
print("  mail list 'probability of the taller group above 0.8': %d people; above 0.5: %d; truth taller group %d" % ((R > 0.8).sum(), (R > 0.5).sum(), g.sum()))
print("  expected number of errors of the 0.5 rule, sum of min(R, 1-R): %.1f; actual errors against hidden group: %d" % (np.minimum(R, 1 - R).sum(), (hard_g != g).sum()))
print("  the two groups' spreads: k-means assumes equal; fitted sd %.2f and %.2f" % tuple(np.sqrt(gm.covariances_.ravel()[order])))

# ---- Question 2: how many starts
X, _ = people_2d()
fin = []
for s in range(30):
    m = GaussianMixture(3, init_params='random_from_data', tol=1e-7, max_iter=1000, random_state=s).fit(X)
    fin.append(m.score(X))
fin = np.array(fin); best = fin.max()
p = (fin > best - 1e-4).mean()
print("\nQ2: 30 random_from_data starts on the 500 people: best %.4f, %d reach it, distinct endings (to 1e-3):" % (best, (fin > best - 1e-4).sum()), sorted(set(float(v) for v in fin.round(3)), reverse=True))
for n in (1, 2, 3, 5):
    print("  probability that at least one of %d starts reaches the best: %.4f" % (n, 1 - (1 - p) ** n))
fk = [GaussianMixture(3, tol=1e-7, max_iter=1000, random_state=s).fit(X).score(X) for s in range(30)]
print("  default k-means start, 30 seeds: %d reach the best (%.4f)" % ((np.array(fk) > best - 1e-4).sum(), max(fk)))
Xw = StandardScaler().fit_transform(load_wine().data)
fw = np.array([GaussianMixture(3, covariance_type='full', init_params='random_from_data', reg_covar=1e-3, tol=1e-6, max_iter=500, random_state=s).fit(Xw).score(Xw) for s in range(12)])
print("  wine, full covariances, K = 3, 12 random_from_data starts: mean log-lik per wine, sorted:", np.sort(fw)[::-1].round(2).tolist())
print("  distinct values within 0.01:", len(set((fw / 0.01).round().astype(int))), "of 12; best %.2f, worst %.2f, ln-lik difference over 178 wines %.0f" % (fw.max(), fw.min(), 178 * (fw.max() - fw.min())))

# ---- Question 3: BIC against AIC
print("\nQ3: n = 500, ln n = %.4f" % np.log(500))
rows = []
for K in range(1, 9):
    m = GaussianMixture(K, n_init=3, random_state=0, tol=1e-5, max_iter=500).fit(X)
    rows.append((K, m.bic(X), m.aic(X), m.score(X) * 500))
print("   K     BIC     AIC   log-lik")
for K, b, a, l in rows: print("  %2d %7.1f %7.1f %9.1f" % (K, b, a, l))
print("  BIC minimum at K =", min(rows, key=lambda t: t[1])[0], "; AIC minimum at K =", min(rows, key=lambda t: t[2])[0])
q = lambda K: (K - 1) + 2 * K + 3 * K
print("  parameters q(K) = 6K - 1: K = 3: %d, K = 4: %d; extra parameters of one more component: %d" % (q(3), q(4), q(4) - q(3)))
print("  gain of log-likelihood needed for one more component: BIC %.1f, AIC %.1f" % ((q(4) - q(3)) * np.log(500) / 2, (q(4) - q(3))))
l3, l4 = rows[2][3], rows[3][3]
print("  actual gain K = 3 -> 4: %.1f" % (l4 - l3))

# ---- Question 4: digits
Xd, yd = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.25, random_state=0, stratify=yd)
pca = PCA(20, random_state=0).fit(Xtr); Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
print("\nQ4: digits, 1347 training and 450 test digits, 20 components")
for K in (10, 20, 30, 40):
    m = GaussianMixture(K, reg_covar=1e-3, n_init=1, random_state=0).fit(Ztr)
    Zs, _ = m.sample(2000); img = pca.inverse_transform(Zs)
    print("  K = %2d: train %.2f test %.2f; parameters %d (1347 samples); draws outside 0..16: %.1f %%" % (K, m.score(Ztr), m.score(Zte), K - 1 + 20 * K + K * 210, 100 * ((img < 0) | (img > 16)).mean()))
m = GaussianMixture(10, reg_covar=1e-3, n_init=3, random_state=0).fit(Ztr)
img = pca.inverse_transform(m.sample(500)[0])
print("  K = 10 (the book's fit): pixel range of 500 draws %.1f to %.1f; share of pixels outside 0..16: %.1f %%; share of pixels exactly 0 in real digits: %.1f %%" % (img.min(), img.max(), 100 * ((img < 0) | (img > 16)).mean(), 100 * (Xd == 0).mean()))
print("  real digit pixel values are integers 0..16: %d distinct values, e.g. share of the value 0, 16: %.2f %%, %.2f %%" % (len(np.unique(Xd)), 100 * (Xd == 0).mean(), 100 * (Xd == 16).mean()))
