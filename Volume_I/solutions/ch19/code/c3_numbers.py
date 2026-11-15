# Numbers for Problem 19C.3: bias and variance of GaussianNB(var_smoothing=0.1) and logistic regression
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
rg = np.random.default_rng(0)
mk = {"NB": lambda: GaussianNB(var_smoothing=0.1),
      "LR": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))}
print(f"{'n':>5s} | {'NB train':>8s} {'NB test':>8s} {'NB sd':>6s} | {'LR train':>8s} {'LR test':>8s} {'LR sd':>6s}")
for n in (40, 80, 160, 320, 640, 1347):
    res = {k: [] for k in mk}
    for r in range(20 if n < 1347 else 1):
        ii = rg.choice(len(ytr), n, replace=False) if n < 1347 else np.arange(len(ytr))
        for k, f in mk.items():
            m = f().fit(Xtr[ii], ytr[ii]); res[k].append((m.score(Xtr[ii], ytr[ii]), m.score(Xte, yte)))
    a = {k: np.array(v) for k, v in res.items()}
    print(f"{n:5d} | {a['NB'][:,0].mean():8.3f} {a['NB'][:,1].mean():8.3f} {a['NB'][:,1].std():6.3f} | {a['LR'][:,0].mean():8.3f} {a['LR'][:,1].mean():8.3f} {a['LR'][:,1].std():6.3f}")
print("number of parameters: GaussianNB", 10 * 64 * 2 + 9, " logistic regression", 10 * 64 + 10)
# the smallest sample: 30 images with all ten digits present (stratified draw of 3 per digit)
rs = np.random.default_rng(1); acc = {k: [] for k in mk}
for r in range(40):
    ii = np.concatenate([rs.choice(np.where(ytr == d)[0], 3, replace=False) for d in range(10)])
    for k, f in mk.items(): acc[k].append(f().fit(Xtr[ii], ytr[ii]).score(Xte, yte))
print("30 images (3 per digit), 40 draws: NB", round(np.mean(acc['NB']), 3), "+-", round(np.std(acc['NB']), 3), " LR", round(np.mean(acc['LR']), 3), "+-", round(np.std(acc['LR']), 3),
      " NB better in", sum(a > b for a, b in zip(acc['NB'], acc['LR'])), "of 40 draws")
