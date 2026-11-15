# Problem 19B.2: sweeping the smoothing constant alpha
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

msgs = ["free prize now", "win free prize", "win free now now",
        "meeting report", "lunch meeting now", "report lunch meeting"]
y = np.array([1, 1, 1, 0, 0, 0])                 # 1 = spam
vec = CountVectorizer()
X = vec.fit_transform(msgs)
q = vec.transform(["free meeting now"])
words = vec.get_feature_names_out()
i_free = list(words).index("free")

print(f"{'alpha':>7s} {'P(spam)':>8s} {'theta(free|ham)':>16s} "
      f"{'theta(free|spam)':>17s} {'log-odds':>9s}")
for a in (0.01, 0.1, 1, 10, 100, 1e4, 1e6):
    m = MultinomialNB(alpha=a).fit(X, y)
    p = m.predict_proba(q)[0, 1]
    th = np.exp(m.feature_log_prob_)     # theta_kj; rows ham, spam
    print(f"{a:7g} {p:8.4f} {th[0, i_free]:16.5f} "
          f"{th[1, i_free]:17.5f} {np.log(p / (1 - p)):9.5f}")
print("uniform value 1/p =", round(1 / len(words), 5))
