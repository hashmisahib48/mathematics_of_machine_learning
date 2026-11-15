# A first project: load, split, fit, predict, score, inspect.
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X, y, names = iris.data, iris.target, iris.target_names
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
print("train, test :", Xtr.shape, Xte.shape)
print("test classes:", np.bincount(yte))
model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(Xtr, ytr)
pred = model.predict(Xte)
print("accuracy    :", round(model.score(Xte, yte), 4))
print("train acc.  :", round(model.score(Xtr, ytr), 4))
print("confusion matrix (rows true, columns predicted):")
print(confusion_matrix(yte, pred))
wrong = np.flatnonzero(pred != yte)
for i in wrong:                       # inspect every mistake
    p = model.predict_proba(Xte[[i]])[0]
    print("sample", i, "features", Xte[i])
    print("  true:", names[yte[i]], " predicted:", names[pred[i]])
    print("  P(setosa, versicolor, virginica):", p.round(3))
new = np.array([[5.0, 3.4, 1.5, 0.2]])
print("new flower  :", names[model.predict(new)[0]])