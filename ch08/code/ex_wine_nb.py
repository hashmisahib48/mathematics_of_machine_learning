# Gaussian naive Bayes on the wine data; scaling does not matter.
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
for name, m in [("raw", GaussianNB()),
                ("standardized",
                 make_pipeline(StandardScaler(), GaussianNB()))]:
    acc = cross_val_score(m, X, y, cv=5)
    print(f"{name:13s} accuracy {acc.mean():.3f}")
g = GaussianNB().fit(X, y)
print("parameters:", g.theta_.size + g.var_.size + 3)
