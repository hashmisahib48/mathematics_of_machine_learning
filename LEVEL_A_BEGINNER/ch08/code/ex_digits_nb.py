# Three naive Bayes models for the handwritten digits.
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

X, y = load_digits(return_X_y=True)
a, b, c, d = train_test_split(X, y, test_size=0.25,
                              random_state=0, stratify=y)
for m in (GaussianNB(), MultinomialNB(), BernoulliNB(binarize=8)):
    acc = m.fit(a, c).score(b, d)
    print(f"{type(m).__name__:14s} accuracy {acc:.3f}")
