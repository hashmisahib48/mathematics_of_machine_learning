import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import (MultinomialNB, ComplementNB,
                                 BernoulliNB)

msgs = ["free prize now", "win free prize", "win free now now",
        "meeting report", "lunch meeting now",
        "report lunch meeting"]
y = np.array([1, 1, 1, 0, 0, 0])             # 1 = spam, 0 = ham
vec = CountVectorizer()                  # black box (Chapter 52)
X = vec.fit_transform(msgs)                  # sparse 6 x 7 counts
print("vocabulary:", list(vec.get_feature_names_out()))

new = vec.transform(["free meeting now", "win prize now"])
models = {"MultinomialNB": MultinomialNB(),  # alpha = 1
          "ComplementNB": ComplementNB(),
          "BernoulliNB": BernoulliNB()}
for name, m in models.items():
    m.fit(X, y)
    p = m.predict_proba(new)[:, 1].round(4)
    print(f"{name:14s} P(spam) = {p}  predicts {m.predict(new)}")

mnb = models["MultinomialNB"]
print("feature_count_:\n", mnb.feature_count_)
print("class_log_prior_  =", mnb.class_log_prior_.round(4))
print("joint log proba of 'free meeting now':",
      mnb.predict_joint_log_proba(new[:1]).round(4))
