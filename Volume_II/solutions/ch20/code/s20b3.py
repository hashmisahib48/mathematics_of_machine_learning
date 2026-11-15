# Problem 20B.3: 5-NN on the wine split with three kinds of scaling
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

X, y = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
models = {"raw": KNeighborsClassifier(),
          "standard": make_pipeline(StandardScaler(),
                                    KNeighborsClassifier()),
          "min-max": make_pipeline(MinMaxScaler(),
                                   KNeighborsClassifier())}
pred = {}
for name, m in models.items():
    m.fit(Xtr, ytr)
    pred[name] = m.predict(Xte)
    print(f"{name:9s} test accuracy {m.score(Xte, yte):.4f}"
          f" = {(pred[name] == yte).sum()} of {len(yte)}")
diff = np.where(pred["standard"] != pred["min-max"])[0]
print("wines on which standard and min-max differ:", diff.tolist())
Zs = StandardScaler().fit_transform(Xtr)
Zm = MinMaxScaler().fit_transform(Xtr)
for name, Z in (("raw", Xtr), ("standard", Zs), ("min-max", Zm)):
    print(f"spread of the columns, {name:9s}:"
          f" {Z.std(0).min():.3f} to {Z.std(0).max():.3f}")
i, j = np.triu_indices(len(Zs), 1)
ds = np.sqrt(((Zs[i] - Zs[j]) ** 2).sum(1))
dm = np.sqrt(((Zm[i] - Zm[j]) ** 2).sum(1))
print("correlation of the two sets of pairwise distances: %.3f"
      % np.corrcoef(ds, dm)[0, 1])
