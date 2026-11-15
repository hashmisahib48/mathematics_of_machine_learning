# Problem 52B.2: learning from a stream with the hashing trick
import sys
import numpy as np
sys.path.append("../../../ch52/code")    # book helpers (run here)
from sklearn.feature_extraction.text import (CountVectorizer,
                                             HashingVectorizer)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from corpus import messages

msgs, y = messages()
Xtr, Xte, ytr, yte = train_test_split(msgs, y, test_size=12,
                                      stratify=y, random_state=0)
perm = np.random.RandomState(0).permutation(len(Xtr))
Xtr = [Xtr[i] for i in perm]; ytr = np.array(ytr)[perm]   # shuffle
hv = HashingVectorizer(n_features=2**12, alternate_sign=False,
                       norm=None)                  # nothing to fit
nb = MultinomialNB()
for b in range(4):                                 # four batches of 9
    s = slice(9 * b, 9 * b + 9)
    nb.partial_fit(hv.transform(Xtr[s]), ytr[s], classes=[0, 1, 2, 3])
    acc = nb.score(hv.transform(Xte), yte)
    print("after %2d messages: accuracy %.4f   per class %s"
          % (9 * (b + 1), acc, np.bincount(ytr[:9 * (b + 1)])))
full = MultinomialNB().fit(hv.transform(Xtr), ytr)
voc = make_pipeline(CountVectorizer(), MultinomialNB()).fit(Xtr, ytr)
print("batch fit, hashed: %.4f" % full.score(hv.transform(Xte), yte))
print("batch fit, vocabulary: %.4f" % voc.score(Xte, yte))
same = np.allclose(full.feature_count_, nb.feature_count_)
print("stream = batch fit?", same, " vocabulary stored:",
      hasattr(hv, "vocabulary_"))
