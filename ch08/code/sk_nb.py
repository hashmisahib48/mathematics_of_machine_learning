# The examples of Section 8.4 with scikit-learn.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB

X = [[2, 6], [3, 8], [4, 7], [6, 3], [7, 5], [8, 4], [7, 4]]
y = ["A", "A", "A", "B", "B", "B", "B"]
g = GaussianNB().fit(X, y)
print("priors", g.class_prior_.round(4), " means", g.theta_.tolist())
print("P at (5, 5):", g.predict_proba([[5, 5]]).round(4))
docs = ["win money now", "free money free prize",
        "meeting now", "lunch money tomorrow"]
labels = ["spam", "spam", "ham", "ham"]
cv = CountVectorizer()
C = cv.fit_transform(docs)                # word counts
print(cv.get_feature_names_out())
m = MultinomialNB(alpha=1.0).fit(C, labels)
z = cv.transform(["free money now"])
print(m.classes_, m.predict_proba(z).round(4))
