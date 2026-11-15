# Problem 7B.3: softmax of large scores, naive and shifted
import numpy as np
from scipy.special import softmax

f = np.array([1000., 1001., 1002.])
with np.errstate(over="ignore", invalid="ignore"):
    e = np.exp(f)                            # e^1000 does not fit
    naive = e / e.sum()                      # Eq. 7.5-1 as written
print("np.exp(f)        :", e)
print("naive            :", naive)

g = f - f.max()                             # Eq. 7.5-2: shift
eg = np.exp(g)
shifted = eg / eg.sum()
print("shifted scores   :", g)
print("exp of shifted   :", eg.round(4), " sum", eg.sum().round(4))
print("shifted softmax  :", shifted.round(4))
print("scipy softmax    :", softmax(f).round(4))
print("equal            :", np.allclose(shifted, softmax(f)))

for t in (float, np.float32):                # where exp overflows
    top = np.log(np.finfo(t).max).round(2)
    print(t.__name__, "overflows above f =", top)
print("softmax of (0, 1, 2)        :", softmax([0., 1, 2]).round(4))
