# Numbers quoted in the model answers to the questions of Chapter 19
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
np.set_printoptions(precision=4, suppress=True)

print("== Q1  conditional independence")
rng = np.random.default_rng(0); n = 20000; y = np.r_[np.zeros(n), np.ones(n)].astype(int)
x1 = rng.normal(size=2 * n) + 2.0 * y; x2 = rng.normal(size=2 * n) + 2.0 * y        # both depend on the class only
print("two features driven by the class only: overall correlation", round(float(np.corrcoef(x1, x2)[0, 1]), 3),
      " within class 0:", round(float(np.corrcoef(x1[y == 0], x2[y == 0])[0, 1]), 3), " within class 1:", round(float(np.corrcoef(x1[y == 1], x2[y == 1])[0, 1]), 3))
Xd, yd = load_digits(return_X_y=True)
img = Xd.reshape(-1, 8, 8)
def adj_corr(sel):
    out = []
    for r in range(8):
        for c in range(7):
            a, b = img[sel, r, c], img[sel, r, c + 1]
            if a.std() > 0 and b.std() > 0: out.append(np.corrcoef(a, b)[0, 1])
    return np.mean(out), len(out)
allm, k_all = adj_corr(np.ones(len(yd), bool)); wc = [adj_corr(yd == k) for k in range(10)]
print(f"digits, horizontally adjacent pixels: mean correlation over all images {allm:.3f} ({k_all} pairs); inside one digit class, averaged over the ten classes {np.mean([w[0] for w in wc]):.3f}")
print("pairs of pixels at distance 4 columns inside classes:", end=" ")
far = []
for k in range(10):
    for r in range(8):
        for c in range(4):
            a, b = img[yd == k, r, c], img[yd == k, r, c + 4]
            if a.std() > 0 and b.std() > 0: far.append(np.corrcoef(a, b)[0, 1])
print(round(float(np.mean(far)), 3))

print("\n== Q2  zero counts (the six messages)")
from sklearn.feature_extraction.text import CountVectorizer
msgs = ["free prize now", "win free prize", "win free now now", "meeting report", "lunch meeting now", "report lunch meeting"]; ys = np.array([1, 1, 1, 0, 0, 0])
vec = CountVectorizer(); Xm = vec.fit_transform(msgs).toarray().astype(float); words = list(vec.get_feature_names_out()); q = vec.transform(["free meeting now"]).toarray()[0]
N = np.array([Xm[ys == k].sum(0) for k in (0, 1)])
print("words:", words); print("N ham :", N[0], N[0].sum(), "\nN spam:", N[1], N[1].sum())
with np.errstate(divide="ignore", invalid="ignore"):
    th0 = N / N.sum(1, keepdims=True); s = np.log(0.5) + np.where(q > 0, q * np.log(th0), 0).sum(1)
print("alpha = 0: log scores of 'free meeting now':", s, " -> joint scores", np.exp(s), " (0/0)")
for a in (1e-3, 0.01, 1.0):
    th = (N + a) / (N + a).sum(1, keepdims=True); sc = np.log(0.5) + (q * np.log(th)).sum(1); print(f"alpha = {a}: theta(free|ham) = {th[0, 0]:.5f}  theta(meeting|spam) = {th[1, 2]:.5f}  P(spam) = {np.exp(sc[1] - np.logaddexp(*sc)):.4f}")
th = (N + 1) / (N + 1).sum(1, keepdims=True)
print("alpha = 1: theta(free|ham) =", round(th[0, 0], 4), "= 1/15; a word seen 3 times in ham:", round(th[0, 2] + 0, 4) if False else round(th[0, 2], 4), " ln ratio 1/15 : 4/15 =", round(float(np.log(4)), 4))

print("\n== Q3  fraud 0.1 per cent")
rng = np.random.default_rng(1); n = 100000; yf = (rng.random(n) < 0.001).astype(int); p = 5
Xf = rng.normal(size=(n, p)) + 1.0 * yf[:, None]; print("fraud cases:", int(yf.sum()), "of", n)
for name, m in (("fit_prior=True", GaussianNB()), ("fit_prior=False (equal priors)", GaussianNB(priors=[0.5, 0.5])), ("priors = (0.99, 0.01)", GaussianNB(priors=[0.99, 0.01])), ("priors = (0.9, 0.1)", GaussianNB(priors=[0.9, 0.1]))):
    pr = m.fit(Xf, yf).predict(Xf); tp = int(((pr == 1) & (yf == 1)).sum()); fp = int(((pr == 1) & (yf == 0)).sum())
    print(f"{name:32s} flagged {int(pr.sum()):6d}  caught {tp:3d} of {int(yf.sum())}  false alarms {fp:6d}  precision {tp / max(pr.sum(), 1):.3f}")
Xt = Xf; mt = GaussianNB().fit(Xf, yf); pp = mt.predict_proba(Xf)[:, 1]; print("largest predicted fraud probability with the true prior:", pp.max().round(3), " samples above 0.5:", int((pp > 0.5).sum()))
print("with equal priors the posterior odds are multiplied by 999:", 999)

print("\n== Q4  naive Bayes against logistic regression on the digits")
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.25, random_state=0, stratify=yd)
for nm, m in (("GaussianNB (default)", GaussianNB()), ("GaussianNB(var_smoothing=0.1)", GaussianNB(var_smoothing=0.1)), ("MultinomialNB", MultinomialNB()), ("logistic (scaled)", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)))):
    m.fit(Xtr, ytr); P_ = m.predict_proba(Xte); print(f"{nm:30s} accuracy {m.score(Xte, yte):.4f}  mean probability of the winning class {P_.max(1).mean():.3f}")
rg = np.random.default_rng(0)
print("test accuracy against training size (mean of 30 random subsets; GaussianNB var_smoothing 0.1 | logistic):")
for n_ in (20, 30, 50, 100, 200, 400, 1347):
    a1, a2 = [], []
    for r in range(30 if n_ < 1347 else 1):
        ii = rg.choice(len(ytr), n_, replace=False) if n_ < 1347 else np.arange(len(ytr))
        if len(np.unique(ytr[ii])) < 10: continue
        a1.append(GaussianNB(var_smoothing=0.1).fit(Xtr[ii], ytr[ii]).score(Xte, yte)); a2.append(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr[ii], ytr[ii]).score(Xte, yte))
    print(f"  n = {n_:5d}: {np.mean(a1):.3f} | {np.mean(a2):.3f}   ({len(a1)} subsets with all ten digits)")
