# Shared setup of Chapter 56: the breast-cancer split of Chapter 22.
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def cancer_split():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.25, random_state=0,
                            stratify=y)


def svc_pipe():
    return make_pipeline(StandardScaler(), SVC())


# the 8 x 9 log grid of Chapter 22 (Volume II): 72 cells
GRID = {"svc__C": 2.0 ** np.arange(-5, 10, 2),
        "svc__gamma": 2.0 ** np.arange(-15, 2, 2)}
