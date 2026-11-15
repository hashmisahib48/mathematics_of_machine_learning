# Case study: the parts of handwritten digits, NMF against PCA
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import NMF, PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      random_state=0, stratify=y)
k = 16
nmf = NMF(k, max_iter=1000, random_state=0).fit(Xtr)
pca = PCA(k).fit(Xtr)
for name, m in (("NMF", nmf), ("PCA", pca)):
    Ztr, Zte = m.transform(Xtr), m.transform(Xte)
    err = np.linalg.norm(Xte - m.inverse_transform(Zte))
    knn = KNeighborsClassifier(5).fit(Ztr, ytr)
    print(f"{name}: test error {err:6.1f}  5-NN accuracy "
          f"{knn.score(Zte, yte):.4f}")
raw = KNeighborsClassifier(5).fit(Xtr, ytr).score(Xte, yte)
print("5-NN on the 64 pixels:", round(raw, 4))
H = nmf.components_
used = (H > 0.1 * H.max(axis=1, keepdims=True)).sum(axis=1)
print("NMF parts: share of zero pixels", (H == 0).mean().round(3),
      " pixels above 10% of the peak, per part:", used)
print("PCA components: share of negative weights",
      (pca.components_ < 0).mean().round(3))
W = nmf.transform(Xte)
print("NMF scores: share of exact zeros", (W == 0).mean().round(3),
      " parts active per image:",
      (W > 0).sum(axis=1).mean().round(1))
