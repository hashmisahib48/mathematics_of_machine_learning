# FeatureAgglomeration on the 8 x 8 digits: 64 pixels -> k groups.
import numpy as np
from sklearn.cluster import FeatureAgglomeration
from sklearn.datasets import load_digits
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)

def accuracy(f=None):               # scaled logistic regression
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000))
    A, B = (Xtr, Xte) if f is None else (f.transform(Xtr),
                                         f.transform(Xte))
    return clf.fit(A, ytr).score(B, yte)

print(f"all 64 pixels: test accuracy {accuracy():.4f}")
grid = grid_to_graph(8, 8)           # pixel neighbours on the grid
print("  k   free    grid")
for k in (4, 8, 16, 32):
    free = FeatureAgglomeration(n_clusters=k).fit(Xtr)
    near = FeatureAgglomeration(n_clusters=k, connectivity=grid)
    acc = accuracy(free), accuracy(near.fit(Xtr))
    print(f"{k:3d}  {acc[0]:.4f}  {acc[1]:.4f}")
