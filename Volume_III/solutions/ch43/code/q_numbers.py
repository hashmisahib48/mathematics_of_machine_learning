# Numbers quoted in the model answers to the questions for discussion of Chapter 43
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../ch43/code')
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import NMF, PCA, FactorAnalysis, FastICA, LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from corpus import make_docs
np.set_printoptions(precision=4, suppress=True)

print("== Q1  PCA and NMF on the digits (split of Section 43.10)")
X, y = load_digits(return_X_y=True)
Xtr, Xte = train_test_split(X, test_size=0.25, random_state=0, stratify=y)[:2]
for k in (4, 10, 16):
    nmf = NMF(k, max_iter=1000, random_state=0).fit(Xtr); pca = PCA(k).fit(Xtr)
    en = np.linalg.norm(Xte - nmf.inverse_transform(nmf.transform(Xte)))
    ep = np.linalg.norm(Xte - pca.inverse_transform(pca.transform(Xte)))
    print(f"k={k:2d}: test error NMF {en:6.1f}  PCA {ep:6.1f}  excess {en-ep:5.1f} ({100*(en/ep-1):4.1f} %)"
          f"  NMF zeros in H {(nmf.components_==0).mean():.3f}  PCA negative weights {(pca.components_<0).mean():.3f}")
nmf = NMF(16, max_iter=1000, random_state=0).fit(Xtr); pca = PCA(16).fit(Xtr)
print("PCA component 1 share of negative weights:", (pca.components_[0] < 0).mean().round(3))
print("one digit (index 0) as PCA scores, first 5:", pca.transform(Xte[:1])[0, :5].round(1), " NMF codes, first 5:", nmf.transform(Xte[:1])[0, :5].round(2))
print("NMF codes of that digit: negative entries", int((nmf.transform(Xte[:1]) < 0).sum()), " PCA scores negative:", int((pca.transform(Xte[:1]) < 0).sum()))

print("\n== Q2  FastICA, two runs on the mixtures of Listing 43.4")
t = np.linspace(0, 8, 2000); S = np.c_[np.sin(2*t), 2*(0.7*t % 1) - 1]; S = (S - S.mean(0)) / S.std(0)
A = np.array([[1.0, 1.0], [0.5, 2.0]]); rng = np.random.RandomState(0); Xm = S @ A.T + 0.05 * rng.standard_normal(S.shape)
E = {}
for rs in range(8):
    f = FastICA(2, random_state=rs); E[rs] = f.fit_transform(Xm)
def signed(Se): return np.corrcoef(Se.T, S.T)[:2, 2:]
for rs in range(8):
    c = signed(E[rs]); order = c.argmax(1)
    print(f"random_state {rs}: signed corr with (sine, saw):", c.round(3).tolist(), " order", "(sine, saw)" if order[0]==0 else "(saw, sine)", " signs", np.sign(c[np.arange(2), order]).astype(int).tolist())
orders = [signed(E[rs]).argmax(1)[0] for rs in range(8)]
print("runs with the sawtooth first:", orders.count(1), " with the sine first:", orders.count(0))
flips = [int((np.sign(signed(E[rs])[np.arange(2), signed(E[rs]).argmax(1)]) < 0).any()) for rs in range(8)]
print("runs in which at least one estimate has the opposite sign to its source:", sum(flips), "of 8")
rs_pair = [rs for rs in range(8) if signed(E[rs]).argmax(1)[0] != signed(E[0]).argmax(1)[0]][:1]
print("first run with the opposite order to run 0:", rs_pair)
print("scale of the estimates (std):", E[0].std(0).round(3), " scale of the sources:", S.std(0).round(3))
for rs in range(8):
    c = np.abs(signed(E[rs])); print(f" run {rs}: worst absolute match {c.max(1).min():.4f}", end=";")
print()
F = FastICA(2, random_state=0, whiten="arbitrary-variance").fit(Xm)
print("whiten='arbitrary-variance': std of the estimates", F.transform(Xm).std(0).round(3))

print("\n== Q3  PCA and factor analysis, one factor")
rng = np.random.RandomState(0); load = np.array([.9, .8, .7, .6, .5, .2])
for name, psi in (("unequal noise", np.array([.2,.2,.2,.2,.2,3.0])), ("equal noise 0.5", np.full(6, 0.5))):
    z = rng.standard_normal((20000, 1)); Xf = z @ load[None] + rng.standard_normal((20000, 6)) * np.sqrt(psi)
    fa = FactorAnalysis(1, random_state=0).fit(Xf); pc = PCA(1).fit(Xf)
    a = fa.components_[0]; b = pc.components_[0]; cosv = abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b)
    print(f"{name}: diag of covariance {np.diag(np.cov(Xf.T)).round(2)}")
    print(f"   FA direction {np.abs(a/np.linalg.norm(a)).round(3)}  PCA direction {np.abs(b).round(3)}  cosine {cosv:.4f}")
Sig = np.outer(load, load) + np.diag([.2,.2,.2,.2,.2,3.0]); w, V = np.linalg.eigh(Sig)
print("exact Sigma = L L^T + Psi, unequal noise: top eigenvector", np.abs(V[:, -1]).round(3), " eigenvalue", w[-1].round(3), " Sigma_66 =", Sig[5, 5])
print("L L^T alone has top direction", (load/np.linalg.norm(load)).round(3), " eigenvalue", (load@load).round(3))

print("\n== Q4  LDA on the invented corpus: size of the problem")
docs, hid = make_docs(0); Xc = CountVectorizer(stop_words="english", min_df=2).fit_transform(docs).toarray()
tr = Xc[:45]; te = Xc[45:]
print("training tokens", int(tr.sum()), " held-out tokens", int(te.sum()), " vocabulary", Xc.shape[1], " words unseen in training that occur in held-out docs:", int(((tr.sum(0) == 0) & (te.sum(0) > 0)).sum()), "of", int((te.sum(0) > 0).sum()))
for K in (1, 4, 8): print(f"K={K}: free numbers in the topic-word table {K*(Xc.shape[1]-1)}")
