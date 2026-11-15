# LDA on an invented corpus (Chapter 52 treats text features)
import numpy as np
from corpus import make_docs
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score as ari

docs, hidden = make_docs(seed=0)            # 60 invented documents
cv = CountVectorizer(stop_words="english", min_df=2)
X = cv.fit_transform(docs)                  # document-word counts
words = np.array(cv.get_feature_names_out())
print("counts", X.shape, " tokens", int(X.sum()))

pure = hidden >= 0                          # 48 pure docs
for it in (10, 100):                        # the default is 10
    lda = LDA(4, max_iter=it, random_state=0).fit(X)
    th = lda.transform(X)                   # each row sums to 1
    score = ari(hidden[pure], th[pure].argmax(axis=1))
    print(f"max_iter {it:3d}: perplexity {lda.bound_:7.2f}, "
          f"ARI {score:.3f}")
for k, row in enumerate(lda.components_):  # pseudo-counts
    top = words[np.argsort(row)[::-1][:6]]
    print(f"topic {k}:", " ".join(top))
print("priors:", lda.doc_topic_prior_, lda.topic_word_prior_)
mixed = np.flatnonzero(~pure)[0]
print("topic mix of mixed document", mixed, ":", th[mixed].round(2))
print("pseudo-counts per topic:",
      lda.components_.sum(axis=1).round(1))
