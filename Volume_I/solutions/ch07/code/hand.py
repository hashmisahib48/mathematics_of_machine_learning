# Hand checks for Problems 7A.1 - 7A.4 (every number quoted in the solutions)
import math
import numpy as np
import scipy.sparse as sp
from scipy.special import softmax
from sklearn.model_selection import train_test_split
np.set_printoptions(precision=4, suppress=True)

print("== 7A.1  matrix B in CSR")
B = np.array([[5, 0, 0, 6, 0], [0, 0, 8, 0, 0], [0, 1, 0, 0, 3]])
S = sp.csr_array(B)
print("data   :", S.data.tolist()); print("indices:", S.indices.tolist()); print("indptr :", S.indptr.tolist())
print("entries per row:", np.diff(S.indptr).tolist(), " running total:", np.cumsum([0] + np.diff(S.indptr).tolist()).tolist())
a, b = S.indptr[2], S.indptr[3]
print("row 2 owns k =", list(range(a, b)), " data", S.data[a:b].tolist(), " columns", S.indices[a:b].tolist())
print("row 2 read back:", S[[2], :].toarray().ravel().tolist(), " stored", S.nnz, "of", B.size, "=", S.nnz / B.size)

print("\n== 7A.2  standardise 0, 2, 2, 8")
x = np.array([0, 2, 2, 8.])
mu = x.mean(); dev = x - mu
print("mean:", mu, " deviations:", dev, " squares:", dev ** 2, " sum:", (dev ** 2).sum())
var = (dev ** 2).mean(); sd = math.sqrt(var); print("variance (n):", var, " sd:", sd)
z = dev / sd; print("z:", z, " sum z:", z.sum(), " sum z^2:", (z ** 2).sum())
print("z of 12:", (12 - mu) / sd)
sd1 = math.sqrt((dev ** 2).sum() / 3); print("with n-1: z =", dev / sd1, " z(12) =", (12 - mu) / sd1)
print("variance (n-1):", (dev ** 2).sum() / 3, " sd (n-1):", math.sqrt((dev ** 2).sum() / 3), " ratio sqrt(n/(n-1)):", math.sqrt(4 / 3))
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(x[:, None]); print("StandardScaler: mean_", sc.mean_, " scale_", sc.scale_, " z", sc.transform(x[:, None]).ravel(), " z(12)", sc.transform([[12.]]).ravel())

print("\n== 7A.3  softmax of (2, 0, -1)")
f = np.array([2., 0, -1]); e = np.exp(f); print("exp:", e, " sum:", e.sum()); print("P:", e / e.sum(), " sum P:", (e / e.sum()).sum(), " argmax:", f.argmax())
g = f + 5; eg = np.exp(g); print("scores + 5:", g, " exp:", eg, " sum:", eg.sum(), " P:", eg / eg.sum())
print("e^5 =", math.exp(5), " ratio of sums:", eg.sum() / e.sum())
print("scipy:", softmax(f), softmax(g))
print("shifted by max: scores", f - f.max(), " exp", np.exp(f - f.max()), " sum", np.exp(f - f.max()).sum())

print("\n== 7A.4  sizes of the two parts")
for n, t in ((150, 0.2), (150, 0.25), (442, 0.25)):
    nt = math.ceil(t * n)
    a_, b_ = train_test_split(np.arange(n), test_size=t, random_state=0)
    print(f"n={n} f={t}: f*n = {t * n}  ceil = {nt}  train = {n - nt}  | train_test_split gives {len(a_)} and {len(b_)}")
print("Python round(110.5) =", round(110.5), " round(37.5) =", round(37.5), " (half to even)")
print("\n== 7B.3  plain Python")
try:
    math.exp(1000)
except OverflowError as e:
    print("math.exp(1000) raises OverflowError:", e)
