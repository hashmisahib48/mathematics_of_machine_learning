# Section 59.7: saving a model with the facts needed to reload it.
import os, platform, tempfile
import joblib
import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
Xtr, Xte, ytr, yte = train_test_split(
    data.data, data.target, test_size=0.25, random_state=0,
    stratify=data.target)
lr = LogisticRegression(max_iter=1000)
final = make_pipeline(StandardScaler(), lr).fit(Xtr, ytr)
meta = {"scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "columns": list(data.feature_names)}
with tempfile.TemporaryDirectory() as folder:
    path = os.path.join(folder, "cancer.joblib")
    joblib.dump({"model": final, "meta": meta}, path)
    size = os.path.getsize(path)
    back = joblib.load(path)
print({k: v for k, v in back["meta"].items() if k != "columns"})
same = back["model"].predict(Xte) == final.predict(Xte)
print("same predictions after loading:", bool(same.all()))
print("file of", size // 1000, "kB holds", len(meta["columns"]),
      "column names and the fitted pipeline")
