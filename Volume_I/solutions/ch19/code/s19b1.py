# Problem 19B.1: class prior and P(spam | "free meeting now")
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

msgs = ["free prize now", "win free prize", "win free now now",
        "meeting report", "lunch meeting now", "report lunch meeting"]
y = np.array([1, 1, 1, 0, 0, 0])                 # 1 = spam
vec = CountVectorizer()
X = vec.fit_transform(msgs)
q = vec.transform(["free meeting now"])

settings = [("class_prior=[0.5, 0.5]", dict(class_prior=[0.5, 0.5])),
            ("class_prior=[0.9, 0.1]", dict(class_prior=[0.9, 0.1])),
            ("class_prior=[0.1, 0.9]", dict(class_prior=[0.1, 0.9])),
            ("fit_prior=False", dict(fit_prior=False))]
for name, kw in settings:
    m = MultinomialNB(**kw).fit(X, y)
    lp = m.class_log_prior_             # ham first, spam second
    print(f"{name:24s} P(spam) = {m.predict_proba(q)[0, 1]:.4f}",
          f"  prior odds {np.exp(lp[1] - lp[0]):.4f}")

# Eq. 19.3-3 by hand: the likelihood part is the same for all priors
m = MultinomialNB().fit(X, y)
x = q.toarray()[0]
llh = x @ m.feature_log_prob_.T      # sum_j x_j ln(theta_kj)
lr = llh[1] - llh[0]
print("log-ratio:", round(lr, 4), " ratio:", round(np.exp(lr), 4))
for pi in (0.5, 0.9, 0.1):                       # pi = P(ham)
    odds = np.exp(lr) * (1 - pi) / pi
    print(f"P(ham) = {pi}: odds {odds:.4f}"
          f" -> P(spam) = {odds/(1+odds):.4f}")
