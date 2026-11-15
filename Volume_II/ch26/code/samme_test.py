# The test: same weights and same predictions as scikit-learn.
import numpy as np
from samme_scratch import MyAdaBoost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split

for name, load, eta in (("breast cancer", load_breast_cancer, 1.0),
                        ("iris, 3 classes", load_iris, 0.5)):
    X, y = load(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)
    mine = MyAdaBoost(30, eta).fit(Xtr, ytr)
    ref = AdaBoostClassifier(n_estimators=30, learning_rate=eta,
                             random_state=0).fit(Xtr, ytr)
    k = len(mine.alphas_)
    same_a = np.allclose(mine.alphas_, ref.estimator_weights_[:k])
    same_p = np.array_equal(mine.predict(Xte), ref.predict(Xte))
    print(f"{name}: {k} rounds, same alphas {same_a}, "
          f"same predictions {same_p}")
    assert same_a and same_p
