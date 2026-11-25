# Scale matters: k-NN on the wine data, raw and standardized.
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
raw = KNeighborsClassifier(n_neighbors=5)
std = make_pipeline(StandardScaler(), KNeighborsClassifier(5))
for name, m in [("raw features", raw), ("standardized", std)]:
    acc = cross_val_score(m, X, y, cv=5)
    print(f"{name:13s} accuracy {acc.mean():.3f}")
