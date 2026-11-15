# Hand checks for Problems 55A.1 and 55A.2 (Eq. 55.3-2 and the GroupKFold rule of §55.3)
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold

# ---- 55A.1: nine samples of class 0 then five of class 1, K = 4
n_c, K = [9, 5], 4
s_c = [0, 9]                                        # start of each class in the sorted list
print("55A.1  a_kc = #{j : s_c <= j < s_c + n_c, j mod K = k-1}")
A = np.zeros((K, 2), int)
for c in range(2):
    pos = list(range(s_c[c], s_c[c] + n_c[c]))
    for k in range(K):
        js = [j for j in pos if j % K == k]
        A[k, c] = len(js)
        print(f"  class {c}, fold {k+1}: positions {js}  -> {len(js)}")
print("  allocation by hand      :", A.tolist(), " fold sizes", A.sum(1).tolist())
y = np.array([0] * 9 + [1] * 5)
sk = [np.bincount(y[te], minlength=2).tolist() for _, te in StratifiedKFold(K).split(np.zeros((14, 1)), y)]
print("  StratifiedKFold(4)      :", sk, " agree:", sk == A.tolist())

# ---- 55A.2: seven patients with 7, 5, 4, 3, 3, 2, 1 rows, K = 3
size = np.array([7, 5, 4, 3, 3, 2, 1]); K = 3
order = np.argsort(size)[::-1]                      # source: argsort then reversed, so ties go to the larger index
print("\n55A.2  processing order (patients, 0-based):", order.tolist(), " sizes", size[order].tolist())
load = [0] * K; mem = [[] for _ in range(K)]
for g in order:
    k = int(np.argmin(load))                        # fold with the fewest rows so far (first one on a tie)
    load[k] += int(size[g]); mem[k].append(int(g))
    print(f"  patient {g} ({size[g]} rows) -> fold {k+1}; loads now {load}")
print("  by hand:", [(sorted(m), int(l)) for m, l in zip(mem, load)])
g = np.repeat(np.arange(7), size)
sk = [(sorted(set(g[te].tolist())), len(te)) for _, te in GroupKFold(K).split(np.zeros((len(g), 1)), groups=g)]
print("  GroupKFold(3)           :", sk, " agree:", sk == [(sorted(m), int(l)) for m, l in zip(mem, load)])
# what happens if ties were taken in the other order (patient 3 before patient 4)?
alt = [0, 1, 2, 3, 4, 5, 6]; load = [0] * K; mem = [[] for _ in range(K)]
for gg in alt:
    k = int(np.argmin(load)); load[k] += int(size[gg]); mem[k].append(gg)
print("  with ties in the other order:", [(m, int(l)) for m, l in zip(mem, load)])

# test folds of 55A.1 as sample numbers (class 0 = samples 0-8, class 1 = samples 9-13)
y = np.array([0] * 9 + [1] * 5)
print("\n55A.1  test folds as sample numbers:", [te.tolist() for _, te in StratifiedKFold(4).split(np.zeros((14, 1)), y)])
