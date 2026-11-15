# Case study: co-clustering 80 invented documents and their words.
import numpy as np
from sklearn.cluster import KMeans, SpectralCoclustering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import consensus_score
from data import TOPICS, corpus

docs, topic, truth = corpus()          # invented for the book
cv = CountVectorizer(stop_words="english", min_df=2)
X = cv.fit_transform(docs)             # documents x words
words = np.array(cv.get_feature_names_out())
print("table", X.shape, "non-zero", X.nnz, "count", int(X.sum()))
for K in (2, 3, 4, 5, 6):
    m = SpectralCoclustering(K, random_state=0).fit(X)
    rl, cl = m.row_labels_, m.column_labels_
    inside = sum(X[rl == k][:, cl == k].sum() for k in range(K))
    print(f"K={K}: ARI {ari(topic, m.row_labels_):.3f},"
          f" {inside / X.sum():.3f} of counts in the blocks")
co = SpectralCoclustering(4, random_state=0).fit(X)
km = KMeans(4, n_init=10, random_state=0).fit(X.toarray())
print("k-means on the rows: ARI", round(ari(topic, km.labels_), 3))
for k in range(4):
    r, c = co.rows_[k], co.columns_[k]
    cnt = np.asarray(X[r][:, c].sum(0)).ravel()
    top = ", ".join(words[c][np.argsort(-cnt)][:3])
    print(f"bicluster {k}: {r.sum()} docs, {c.sum()} words: {top}")
rows = np.array([topic == t for t in range(4)])
cols = np.array([np.isin(words, sorted(truth[n])) for n in TOPICS])
print("consensus score:",
      round(consensus_score(co.biclusters_, (rows, cols)), 3))
for w in ("star", "team", "market", "season"):
    j = list(words).index(w)
    counts = [int(X[topic == t][:, j].sum()) for t in range(4)]
    print(w, "-> bicluster", co.column_labels_[j], "counts", counts)
