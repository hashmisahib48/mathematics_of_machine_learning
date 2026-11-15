# Section 52.11: latent semantic analysis of the 60 documents.
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from corpus import make_docs

docs, topic = make_docs(seed=0)        # topic -1: two topics mixed
X = TfidfVectorizer(stop_words="english").fit_transform(docs)
print("documents x words:", X.shape, " non-zero share:",
      round(X.nnz / np.prod(X.shape), 4))

S = cosine_similarity(X)               # unit rows: S = X X^T
np.fill_diagonal(S, -1)
nn = S.argmax(axis=1)                  # most similar other document
pure = topic >= 0
print("nearest document has the same topic:",
      np.mean(topic[nn][pure] == topic[pure]).round(4))

svd = TruncatedSVD(n_components=2, random_state=0)
Z = svd.fit_transform(X)               # LSA, Chapter 42
print("variance explained:", svd.explained_variance_ratio_.round(3))
km = KMeans(n_clusters=4, n_init=10, random_state=0)
lab = km.fit_predict(Z[pure])          # mixed documents left out
print("4 means on LSA: adjusted Rand index",
      round(adjusted_rand_score(topic[pure], lab), 3))
