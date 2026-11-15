# Section 53.5: selecting on all rows before cross-validation.
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

n, p, k = 60, 1000, 10          # pure noise: y has nothing to do with X
leaky, honest = [], []
for seed in range(30):
    rng = np.random.RandomState(seed)
    X, y = rng.standard_normal((n, p)), np.arange(n) % 2
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    clf = LogisticRegression()
    Xk = SelectKBest(k=k).fit_transform(X, y)         # all rows: leak
    leaky.append(cross_val_score(clf, Xk, y, cv=cv).mean())
    pipe = make_pipeline(SelectKBest(k=k), clf)       # refit per fold
    honest.append(cross_val_score(pipe, X, y, cv=cv).mean())
print("chance level              0.500")
print("selected before CV   %.3f +- %.3f" % (np.mean(leaky),
                                              np.std(leaky)))
print("selected inside CV   %.3f +- %.3f" % (np.mean(honest),
                                              np.std(honest)))
