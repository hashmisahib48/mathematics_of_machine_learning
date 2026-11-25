# The worked examples of Section 8.4.
from naive_bayes import GaussianNB, MultinomialNB

X = [[2, 6], [3, 8], [4, 7], [6, 3], [7, 5], [8, 4], [7, 4]]
y = ["A", "A", "A", "B", "B", "B", "B"]
g = GaussianNB().fit(X, y)
for c in g.classes_:
    print(c, "means", g.theta_[c], "variances",
          [round(v, 4) for v in g.var_[c]])
print("log joint at (5, 5):",
      [round(g._joint([5, 5], c), 4) for c in g.classes_])
P = g.predict_proba([[5, 5]])[0]
print("P(A), P(B):", [round(p, 4) for p in P])
# counts of: free lunch meeting money now prize tomorrow win
X = [[0, 0, 0, 1, 1, 0, 0, 1], [2, 0, 0, 1, 0, 1, 0, 0],
     [0, 0, 1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0, 1, 0]]
y = ["spam", "spam", "ham", "ham"]
m = MultinomialNB(alpha=1.0).fit(X, y)
z = [[1, 0, 0, 1, 1, 0, 0, 0]]          # "free money now"
print("P(ham), P(spam):", [round(p, 4) for p in m.predict_proba(z)[0]])
print("class:", m.predict(z)[0])
