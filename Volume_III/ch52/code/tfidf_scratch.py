# Section 52.8: tf-idf from scratch, checked against scikit-learn.
import re
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from corpus import messages

class TfidfScratch:
    pat = re.compile(r"(?u)\b\w\w+\b")      # the default tokens

    def _count(self, docs):
        X = np.zeros((len(docs), len(self.vocab_)))
        for i, d in enumerate(docs):
            for w in self.pat.findall(d.lower()):
                if w in self.vocab_:    # unseen words dropped
                    X[i, self.vocab_[w]] += 1
        return X

    def fit(self, docs):
        words = sorted({w for d in docs
                        for w in self.pat.findall(d.lower())})
        self.vocab_ = {w: j for j, w in enumerate(words)}
        df = (self._count(docs) > 0).sum(axis=0)
        n = len(docs)
        self.idf_ = np.log((1 + n) / (1 + df)) + 1   # Eq. 52.4-2
        return self

    def transform(self, docs):
        W = self._count(docs) * self.idf_            # Eq. 52.4-1
        W /= np.linalg.norm(W, axis=1, keepdims=True)  # Eq. 52.4-3
        return sparse.csr_matrix(W)

msgs, y = messages()
mine = TfidfScratch().fit(msgs[::2])        # fit on every other one
ref = TfidfVectorizer().fit(msgs[::2])
assert mine.vocab_ == ref.vocabulary_
assert np.allclose(mine.idf_, ref.idf_)
A, B = mine.transform(msgs), ref.transform(msgs)  # all 48 messages
print("vocabulary:", len(mine.vocab_), "words")
print("largest difference:", abs(A - B).max())
print("unit rows:", np.allclose(norm(A, axis=1), 1.0))
