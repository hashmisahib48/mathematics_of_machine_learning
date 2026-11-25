# Three classes: multinomial logistic regression on the iris flowers.
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
for C in (0.01, 0.1, 1, 10, 100):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C))
    acc = cross_val_score(m, X, y, cv=5).mean()
    print(f"C = {C:6}: accuracy {acc:.3f}")
m = make_pipeline(StandardScaler(), LogisticRegression(C=10)).fit(X, y)
print("coef_ shape", m[-1].coef_.shape)
print("flower 70:", X[70], "P =", m.predict_proba(X[[70]]).round(3))
