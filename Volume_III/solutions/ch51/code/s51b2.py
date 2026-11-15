# Problem 51B.2: draws from the posterior against predictions
import sys, warnings
import numpy as np
sys.path.append("../../../ch51/code")   # book helpers (run here)
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from holes import split, punch, HOLE
warnings.simplefilter("ignore", ConvergenceWarning)   # 10 rounds

Xtr, Xte, ytr, yte = split()
A = punch(Xtr, "MCAR", 0.2, 0)
M = np.isnan(A)

def error(F):                            # as in Problem 51B.1
    d = [np.sqrt(np.mean((F[M[:, j], j] - Xtr[M[:, j], j]) ** 2))
         / Xtr[:, j].std() for j in HOLE]
    return np.mean(d)

def spread(F):                           # var(filled) / var(true)
    return np.mean([F[:, j].var() / Xtr[:, j].var() for j in HOLE])

plain = IterativeImputer(random_state=0).fit_transform(A)
draws = [IterativeImputer(sample_posterior=True, random_state=s)
         .fit_transform(A) for s in range(5)]
e = [error(F) for F in draws]
print("plain fit        %.3f" % error(plain))
for s in range(5):
    print("draw, seed %d     %.3f" % (s, e[s]))
print("mean of draws    %.3f" % np.mean(e))
print("average of draws %.3f" % error(np.mean(draws, axis=0)))
print("spread: plain %.3f  draw 0 %.3f"
      % (spread(plain), spread(draws[0])))
print("one draw / plain   = %.3f  (sqrt 2 = %.3f)"
      % (np.mean(e) / error(plain), np.sqrt(2)))
print("average / plain    = %.3f  (sqrt 1.2 = %.3f)"
      % (error(np.mean(draws, axis=0)) / error(plain), np.sqrt(1.2)))
