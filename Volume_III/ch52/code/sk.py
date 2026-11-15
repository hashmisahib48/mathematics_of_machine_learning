# Section 52.6 by hand (NumPy) and in scikit-learn.
import numpy as np
from sklearn.feature_extraction.text import (CountVectorizer,
    TfidfTransformer, TfidfVectorizer)

docs = ["bake the bread", "bake the cake bake", "cake and the cake"]
cv = CountVectorizer()
X = cv.fit_transform(docs)                  # counts, sparse
print("terms  ", list(cv.get_feature_names_out()))
print("counts ", X.toarray().tolist(), type(X).__name__)

n = X.shape[0]
df = (X.toarray() > 0).sum(axis=0)          # document frequency
idf = np.log((1 + n) / (1 + df)) + 1        # Eq. 52.4-2
W = X.toarray() * idf                       # Eq. 52.4-1
Z = W / np.linalg.norm(W, axis=1, keepdims=True)   # Eq. 52.4-3
print("df     ", df.tolist(), " idf", idf.round(4).tolist())
print("weights", W.round(3).tolist())
print("unit   ", Z.round(3).tolist())
print("cosines", (Z @ Z.T)[np.triu_indices(3, 1)].round(3))

tt = TfidfTransformer().fit(X)              # from the counts
tv = TfidfVectorizer().fit(docs)            # from the raw text
assert np.allclose(tt.idf_, idf)
assert np.allclose(tt.transform(X).toarray(), Z)
assert np.allclose(tv.transform(docs).toarray(), Z)
print("hand, TfidfTransformer and TfidfVectorizer agree")
