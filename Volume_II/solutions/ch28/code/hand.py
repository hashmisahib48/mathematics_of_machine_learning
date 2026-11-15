# Hand checks for Problems 28A.1 - 28A.3 (every number quoted in the solutions)
import numpy as np
np.set_printoptions(precision=4, suppress=True)

print("== 28A.1  x = 3, 5, 6, 8, 9, 12, 14, 20; B = 4")
x = np.array([3, 5, 6, 8, 9, 12, 14, 20.]); n = len(x)
for q in (0.25, 0.5, 0.75):
    k = n * q                                  # position n*q in the sorted list
    print(f"q={q}: n*q = {k}  integer: {k == int(k)}  -> mean of x[{int(k)}], x[{int(k)+1}] (1-based) =",
          (x[int(k) - 1] + x[int(k)]) / 2)
thr = np.percentile(x, [25, 50, 75], method="averaged_inverted_cdf")
print("thresholds:", thr, " bins (count of thresholds below x):", np.searchsorted(thr, x))
print("bin sizes:", np.bincount(np.searchsorted(thr, x)))
print("numpy default (linear) percentiles instead:", np.percentile(x, [25, 50, 75]))
w = (x.max() - x.min()) / 4; ed = x.min() + w * np.arange(1, 4)
print("equal-width: width", w, " edges", ed, " bins", np.searchsorted(ed, x), " sizes", np.bincount(np.searchsorted(ed, x)))

print("\n== 28A.2  G = (3,-1,-4,2), H = (2,2,2,2), lambda = 1")
G = np.array([3, -1, -4, 2.]); H = np.array([2, 2, 2, 2.]); lam = 1.0
GL, HL = np.cumsum(G)[:-1], np.cumsum(H)[:-1]; GR, HR = G.sum() - GL, H.sum() - HL
print("G =", G.sum(), " H =", H.sum())
print("GL", GL, "HL", HL, "GR", GR, "HR", HR)
print("GL^2/(HL+lam):", GL**2 / (HL + lam)); print("GR^2/(HR+lam):", GR**2 / (HR + lam))
gain = GL**2 / (HL + lam) + GR**2 / (HR + lam) - G.sum()**2 / (H.sum() + lam)
print("gain:", gain, " best after bin", int(gain.argmax()))
print("exact:", [3 + 9/7, 1.6, 4/7 + 4/3])
k = int(gain.argmax()); print("leaf values:", -GL[k] / (HL[k] + lam), -GR[k] / (HR[k] + lam))
print("same cuts, lambda = 0:", GL**2 / HL + GR**2 / HR)

print("\n== 28A.3  histogram subtraction")
Gp = np.array([2.5, -1.5, 0.5, -0.5]); Hp = np.array([1.5, 1.2, 1.0, 1.3])
Gs = np.array([1.0, -0.5, 0.0, -0.5]); Hs = np.array([0.6, 0.4, 0.0, 0.5])
Gl, Hl = Gp - Gs, Hp - Hs
print("G larger:", Gl, " H larger:", Hl, " totals:", Gl.sum(), Hl.sum())
print("parent totals:", Gp.sum(), Hp.sum(), " smaller totals:", Gs.sum(), Hs.sum())
print("leaf value larger child, lambda = 1:", -Gl.sum() / (Hl.sum() + 1.0))
print("leaf value smaller child, lambda = 1:", -Gs.sum() / (Hs.sum() + 1.0), " parent as one leaf:", -Gp.sum() / (Hp.sum() + 1.0))
print("gain of the split (parent histogram ... as leaves):",
      Gs.sum()**2/(Hs.sum()+1) + Gl.sum()**2/(Hl.sum()+1) - Gp.sum()**2/(Hp.sum()+1))
