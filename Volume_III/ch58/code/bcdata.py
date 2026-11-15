# Breast-cancer split of the volume (426 training, 143 test rows).
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def load():
    d = load_breast_cancer()
    Xa, Xb, ya, yb = train_test_split(
        d.data, d.target, test_size=0.25, random_state=0,
        stratify=d.target)
    return Xa, Xb, ya, yb, d.feature_names
