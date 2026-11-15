# Breast cancer: SVC and 5-NN behind each scaler (Section 49.9).
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (train_test_split,
    cross_val_score)
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing as pp
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
scalers = {
    "none": None, "StandardScaler": pp.StandardScaler(),
    "MinMaxScaler": pp.MinMaxScaler(),
    "MaxAbsScaler": pp.MaxAbsScaler(),
    "RobustScaler": pp.RobustScaler(),
    "PowerTransformer": pp.PowerTransformer(),
    "QuantileTransformer": pp.QuantileTransformer(
        n_quantiles=100, output_distribution="normal"),
    "Normalizer (rows)": pp.Normalizer(),
}
print(f"{'scaler':>19}{'SVC test':>9}{'SVC cv':>8}{'5-NN':>10}")
for name, sc in scalers.items():
    row = []
    for est in (SVC(), KNeighborsClassifier(5)):
        pipe = make_pipeline(sc, est) if sc else est
        cv = cross_val_score(pipe, Xtr, ytr, cv=5).mean()
        row += [pipe.fit(Xtr, ytr).score(Xte, yte), cv]
    print(f"{name:>19}{row[0]:9.4f}{row[1]:8.4f}{row[2]:10.4f}")
