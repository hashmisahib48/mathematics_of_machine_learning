# Problem 49B.1: scaling the sparse digit images
import numpy as np
from scipy import sparse
from sklearn.datasets import load_digits
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler,
                                   RobustScaler, StandardScaler)

X = load_digits().data / 16.0              # 1797 x 64, in [0, 1]
S = sparse.csr_matrix(X)
share = lambda M: M.nnz / np.prod(M.shape)   # non-zero share
print(f"{'raw':36s}{share(S):8.4f}")
for t in (MaxAbsScaler(), StandardScaler(with_mean=False),
          RobustScaler(with_centering=False)):
    print(f"{str(t):36s}{share(t.fit_transform(S)):8.4f}")
dense = StandardScaler().fit_transform(X)  # centred: zeros are filled
print(f"{'StandardScaler() on the dense array':36s}"
      f"{(dense != 0).mean():8.4f}")

for t in (StandardScaler(), MinMaxScaler()):   # the two that refuse
    try:
        t.fit(S)
    except Exception as e:
        print(f"{type(t).__name__}: {type(e).__name__}:")
        print("   ", str(e)[:56])

empty = np.flatnonzero(S.getnnz(axis=0) == 0)  # columns with no ink
print("all-zero columns:", empty)
for t in (StandardScaler(with_mean=False), MaxAbsScaler(),
          RobustScaler(with_centering=False)):
    print(f"  scale_, {type(t).__name__}:", t.fit(S).scale_[empty])
rs = RobustScaler(with_centering=False).fit(S).scale_
mostly0 = S.getnnz(axis=0) < 0.25 * S.shape[0]
print("RobustScaler columns with scale_ == 1:", (rs == 1).sum())
print("columns with more than 75% zeros:     ", mostly0.sum())
ss = StandardScaler(with_mean=False).fit_transform(S).toarray()
print("no centring: sd of columns 1, 10, 20:",
      ss.std(0)[[1, 10, 20]].round(3))
print("             mean of column 10:", ss[:, 10].mean().round(3))
