# OutputCodeClassifier on the digits: code book and decoding.
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OutputCodeClassifier
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

base = LogisticRegression(max_iter=300)
ecoc = OutputCodeClassifier(base, code_size=2, random_state=2)
ecoc.fit(Xtr, ytr)
M = ecoc.code_book_
print("bits:", len(ecoc.estimators_), " code book shape:", M.shape)
for k in range(3):                         # first three code words
    print(k, "".join("+" if v > 0 else "-" for v in M[k]))
ham = (M[:, None, :] != M[None, :, :]).sum(axis=2)
print("smallest Hamming distance:",
      ham[np.triu_indices(10, 1)].min())

i = 0                                      # one test digit, by hand
s = np.array([e.decision_function(Xte[i:i + 1])[0]
              for e in ecoc.estimators_])
d = np.sqrt(((s - M) ** 2).sum(axis=1))    # Eq. 33.3-10
print("distances to the ten code words:", d.round(1))
p = ecoc.predict(Xte[i:i + 1])[0]
print("nearest:", d.argmin(), " predict:", p, " true:", yte[i])
print("test accuracy:", round(ecoc.score(Xte, yte), 4))
