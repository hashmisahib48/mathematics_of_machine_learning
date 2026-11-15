# One table in three containers, and what an estimator refuses.
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

A = np.array([[0, 0, 3, 0, 0],
              [1, 0, 0, 0, 2],
              [0, 0, 0, 0, 0],
              [0, 4, 0, 5, 0]])
S = sparse.csr_array(A)             # store only the non-zeros
print("array   :", A.shape, A.dtype)
print("data    :", S.data)
print("indices :", S.indices)
print("indptr  :", S.indptr)
print("stored  :", S.nnz, "of", A.size, "entries")
df = pd.DataFrame({"km": [1.0, 2.0, 3.0], "lights": [0, 2, 1]})
X = df.to_numpy()                   # DataFrame -> 2-D array
print("DataFrame", df.shape, "-> array", X.shape, X.dtype)
col = df["km"].to_numpy()           # one column is 1-D ...
print("column", col.shape, "-> reshaped", col.reshape(-1, 1).shape)
try:
    LinearRegression().fit([[1.0], [np.nan]], [1.0, 2.0])
except ValueError as err:
    print("ValueError:", str(err).splitlines()[0])
try:
    StandardScaler().fit(S)         # centring would destroy zeros
except ValueError as err:
    print("ValueError:", str(err).split(":")[0])
keep = StandardScaler(with_mean=False).fit_transform(S)
print("with_mean=False, still sparse:", sparse.issparse(keep))