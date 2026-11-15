# Problem 52B.1: sparse against dense storage of the count matrix
import sys
sys.path.append("../../../ch52/code")    # book helpers (run here)
from sklearn.feature_extraction.text import CountVectorizer
from corpus import messages

msgs, y = messages()
print("range   shape     nnz  share   CSR bytes  dense bytes  ratio")
for rng in ((1, 1), (1, 2)):
    X = CountVectorizer(ngram_range=rng).fit_transform(msgs)
    csr = X.data.nbytes + X.indices.nbytes + X.indptr.nbytes
    dense = X.shape[0] * X.shape[1] * X.dtype.itemsize
    share = X.nnz / (X.shape[0] * X.shape[1])
    print("%-6s %-9s %4d %5.2f%% %9d %12d %6.1f" % (rng, X.shape,
          X.nnz, 100 * share, csr, dense, dense / csr))
print("dtype", X.dtype, " index dtype", X.indices.dtype)
