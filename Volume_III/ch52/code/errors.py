# Two errors of this chapter, raised on purpose.
from sklearn.feature_extraction.text import (CountVectorizer,
                                             HashingVectorizer)
from sklearn.naive_bayes import MultinomialNB
from corpus import messages

msgs, y = messages()
try:                         # signed hashing, counts model
    X = HashingVectorizer(n_features=2**10).transform(msgs)
    MultinomialNB().fit(X, y)
except ValueError as e:
    print("ValueError:", e)
try:                         # one string, not a list
    CountVectorizer().fit_transform("bake the bread")
except ValueError as e:
    print("ValueError:", e)
