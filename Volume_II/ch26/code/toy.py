# Three rounds of AdaBoost by hand: six apples, one feature.
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6.])           # sugar reading
y = np.array([-1, -1, -1, 1, -1, 1])        # +1 = judged sweet


def best_stump(w):
    """Threshold and sign with the smallest weighted error."""
    best = (np.inf, None, None)
    for t in (x[:-1] + x[1:]) / 2:
        for s in (1, -1):                   # s = answer right of t
            h = np.where(x <= t, -s, s)
            err = w[h != y].sum()           # Eq. 26.3-1
            if err < best[0] - 1e-12:
                best = (err, t, s)
    return best


w, F = np.full(6, 1 / 6), np.zeros(6)
print("round  threshold sign  err    alpha")
for m in (1, 2, 3):
    err, t, s = best_stump(w)
    h = np.where(x <= t, -s, s)
    alpha = np.log((1 - err) / err)         # Eq. 26.3-2, K = 2
    w = w * np.exp(alpha * (h != y))        # Eq. 26.3-3
    w = w / w.sum()
    F += alpha * h                          # Eq. 26.3-4
    print(f"{m:5d}  {t:8.1f} {s:4d}  {err:.4f} {alpha:.4f}")
    print("   weights", np.round(w, 4))
print("F(x)      ", np.round(F, 3))
print("sign F    ", np.sign(F).astype(int), " y", y)
