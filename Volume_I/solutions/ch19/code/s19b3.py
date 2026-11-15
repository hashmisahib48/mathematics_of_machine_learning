# Problem 19B.3: MultinomialNB.partial_fit on five digit batches
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

X, y = load_digits(return_X_y=True)        # 1797 images, 64 pixels
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      random_state=0, stratify=y)
batches = np.array_split(np.arange(len(ytr)), 5)
print("batch sizes:", [len(b) for b in batches])

stream = MultinomialNB()
for t, idx in enumerate(batches, 1):
    # the classes must be named in the first call
    stream.partial_fit(Xtr[idx], ytr[idx], classes=np.arange(10))
    n_seen = int(stream.class_count_.sum())
    print(f"after batch {t}: n_seen = {n_seen:4d}"
          f"  test accuracy = {stream.score(Xte, yte):.4f}")

full = MultinomialNB().fit(Xtr, ytr)
fs, ff = stream.feature_log_prob_, full.feature_log_prob_
print("feature_log_prob_ equal:", np.allclose(fs, ff))
print("largest difference    :", np.abs(fs - ff).max())
print("class_log_prior_ equal :", np.allclose(stream.class_log_prior_,
                                              full.class_log_prior_))
print(f"accuracy of one call to fit: {full.score(Xte, yte):.4f}")
# the same stream in a different order gives the same model
order = np.random.default_rng(1).permutation(len(ytr))
other = MultinomialNB()
for idx in np.array_split(order, 5):
    other.partial_fit(Xtr[idx], ytr[idx], classes=np.arange(10))
same = np.allclose(other.feature_log_prob_, ff)
print("shuffled stream equal  :", same)
