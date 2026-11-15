# Hand calculations of the Chapter 52 solutions: Problems 52A.1, 52A.2 and 52B.1 (sizes)
import os, sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../ch52/code")
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.utils import murmurhash3_32
from corpus import messages

print("=== 52A.1  tf-idf with the other options")
docs = ["tea tea milk", "milk sugar", "tea sugar sugar sugar", "tea"]
cv = CountVectorizer(); X = cv.fit_transform(docs).toarray()
print("vocabulary", cv.get_feature_names_out().tolist()); print("counts\n", X)
df = (X > 0).sum(0); n = len(docs); print("df", df.tolist(), " n =", n)
for name, idf in (("smooth  ln((1+n)/(1+df))+1", np.log((1 + n) / (1 + df)) + 1), ("plain   ln(n/df)+1", np.log(n / df) + 1)):
    print(name, idf.round(4).tolist())
idf_s = np.log((1 + n) / (1 + df)) + 1; idf_p = np.log(n / df) + 1
w = X[0] * idf_s; print("default: weights", w.round(4).tolist(), " length %.4f" % np.linalg.norm(w), " unit row", (w / np.linalg.norm(w)).round(4).tolist())
t = np.where(X[0] > 0, 1 + np.log(np.maximum(X[0], 1)), 0); print("sublinear tf", t.round(4).tolist())
w2 = t * idf_p; print("sublinear, plain idf: weights", w2.round(4).tolist(), " length %.4f" % np.linalg.norm(w2), " unit row", (w2 / np.linalg.norm(w2)).round(4).tolist())
Xc = cv.transform(docs)
print("TfidfTransformer default row 1:", TfidfTransformer().fit_transform(Xc)[0].toarray().round(4).tolist())
print("TfidfTransformer(smooth_idf=False, sublinear_tf=True) row 1:", TfidfTransformer(smooth_idf=False, sublinear_tf=True).fit_transform(Xc)[0].toarray().round(4).tolist())
print("share of the weight of 'tea' (unit rows): default %.3f, sublinear %.3f" % ((w / np.linalg.norm(w))[2], (w2 / np.linalg.norm(w2))[2]))

print("\n=== 52A.2  expected occupied buckets, V = 216")
msgs, y = messages(); V = len(CountVectorizer().fit(msgs).vocabulary_)
words = CountVectorizer().fit(msgs).get_feature_names_out()
hs = np.array([murmurhash3_32(w_, seed=0, positive=False) for w_ in words]).astype(np.int64)
for m in (64, 1024):
    p0 = (1 - 1 / m) ** V; occ = m * (1 - p0); obs = len(set(np.abs(hs) % m))
    print("m = %4d: (1-1/m)^V = %.4f  empty buckets m*(1-1/m)^V = %.2f  occupied = %.2f  words without own bucket = %.2f   observed occupied %d, lost %d   V^2/(2m) = %.2f"
          % (m, p0, m * p0, occ, V - occ, obs, V - obs, V * V / (2 * m)))
from sklearn.feature_extraction.text import HashingVectorizer
for m in (64, 1024):
    Z = HashingVectorizer(n_features=m, norm=None, alternate_sign=False).transform(words)
    print("HashingVectorizer(n_features=%d) on the 216 words: occupied columns %d" % (m, len(set(Z.indices))))
print("ln(1-1/64) = %.6f, 216 x that = %.4f, exp = %.4f" % (np.log(1 - 1 / 64), 216 * np.log(1 - 1 / 64), np.exp(216 * np.log(1 - 1 / 64))))

print("\n=== 52B.1  bytes by hand")
tok = CountVectorizer().build_analyzer(); T = [len(tok(s)) for s in msgs]
print("tokens", sum(T), " per message mean %.2f" % np.mean(T))
for r in ((1, 1), (1, 2)):
    cvr = CountVectorizer(ngram_range=r); X = cvr.fit_transform(msgs); nnz = X.nnz; nr, V_ = X.shape
    print(r, "shape", X.shape, "nnz", nnz, " itemsize", X.dtype.itemsize, " index itemsize", X.indices.dtype.itemsize,
          " csr bytes = %d*(%d+%d) + %d*%d = %d  (measured %d)" % (nnz, X.dtype.itemsize, X.indices.dtype.itemsize, X.indptr.dtype.itemsize, nr + 1,
          nnz * (X.dtype.itemsize + X.indices.dtype.itemsize) + X.indptr.dtype.itemsize * (nr + 1), X.data.nbytes + X.indices.nbytes + X.indptr.nbytes),
          " dense %d*%d*%d = %d" % (nr, V_, X.dtype.itemsize, nr * V_ * X.dtype.itemsize))
bi = [len(CountVectorizer(ngram_range=(2, 2)).build_analyzer()(s)) for s in msgs]
print("bigram tokens", sum(bi), "= tokens - messages =", sum(T) - len(msgs), "; non-zero cells of bigram-only matrix", CountVectorizer(ngram_range=(2, 2)).fit_transform(msgs).nnz)

print("\n=== instructor remarks")
# slip in 52A.1: forgetting the '+ 1' of the idf in the second variant
idf0 = np.log(n / df); w0 = X[0] if False else None
Xc2 = CountVectorizer().fit_transform(docs).toarray(); t1 = np.where(Xc2[0] > 0, 1 + np.log(np.maximum(Xc2[0], 1)), 0) * np.log(4 / df)
print("52A.1 slip, ln(n/df) without +1: weights", t1.round(4).tolist(), "unit row", (t1 / np.linalg.norm(t1)).round(4).tolist())
# standard deviation of the number of occupied buckets (52A.2)
V = 216
for m in (64, 1024):
    e0 = (1 - 1 / m) ** V; e1 = (1 - 2 / m) ** V; mean = m * (1 - e0)
    var = m * (m - 1) * e1 + m * e0 - (m * e0) ** 2
    print("m = %4d: sd of the number of empty buckets %.2f (occupied: mean %.2f)" % (m, np.sqrt(var), mean))
# 52B.1 ratio from the density
for r in ((1, 1), (1, 2)):
    Xr = CountVectorizer(ngram_range=r).fit_transform(msgs); d = Xr.nnz / np.prod(Xr.shape)
    print(r, "density %.4f, (2/3)/density = %.1f" % (d, (2 / 3) / d), " data array alone: %d bytes" % Xr.data.nbytes)
print("vocabulary growth (1,1) -> (1,2): x%.2f   non-zero cells x%.2f" % (575 / 216, 838 / 419))
