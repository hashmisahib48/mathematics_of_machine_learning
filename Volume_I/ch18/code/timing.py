# Fitting time on 100 000 samples, 50 features, 3 classes (not a book listing; times vary by machine).
import time
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
rng = np.random.default_rng(0)
X = rng.normal(size=(100000, 50)); y = rng.integers(0, 3, 100000)
for name, m in (("LDA svd", LDA()), ("LDA lsqr", LDA(solver="lsqr")), ("QDA", QDA())):
    t = time.perf_counter(); m.fit(X, y); print(f"{name}: {time.perf_counter() - t:.2f} s")
