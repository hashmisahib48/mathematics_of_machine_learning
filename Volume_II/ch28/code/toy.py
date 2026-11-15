# Twelve visits: seconds on the page, and did the visitor buy?
import numpy as np

x = np.array([8, 15, 22, 31, 40, 47, 58, 66, 73, 85, 92, 110.])
y = np.array([0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1])
B, lam, eta = 4, 0.0, 0.5       # bins, l2_regularization, shrinkage

# Binning (Eq. 28.3-2): quantile thresholds; bin = those below x.
thr = np.percentile(x, np.linspace(0, 100, B + 1)[1:-1],
                    method="averaged_inverted_cdf")
b = np.searchsorted(thr, x)
print("thresholds", thr, " bins", b)

def stump(F):
    p = 1 / (1 + np.exp(-F))
    g, h = p - y, p * (1 - p)                       # Eq. 28.3-4
    G, H = np.bincount(b, g, B), np.bincount(b, h, B)   # Eq. 28.3-7
    print(" p", p.round(4)[[0, 3, 6]], " g", g.round(3)[[0, 3, 6]])
    print(" G", G.round(4), " H", H.round(4))
    GL, HL = np.cumsum(G)[:-1], np.cumsum(H)[:-1]   # left sums
    GR, HR = G.sum() - GL, H.sum() - HL
    gain = GL**2 / (HL + lam) + GR**2 / (HR + lam)  # Eq. 28.3-9
    k = int(np.argmax(gain))
    print(" gain", gain.round(4), " best: x <=", thr[k])
    return -GL[k] / (HL[k] + lam), -GR[k] / (HR[k] + lam), k

F = np.full(12, np.log(y.mean() / (1 - y.mean())))
print("start: raw score %.4f" % F[0])
for rnd in (1, 2):
    print("round", rnd)
    vL, vR, k = stump(F)                            # Eq. 28.3-10
    print(" leaf values %.4f %.4f, times eta %.4f %.4f"
          % (vL, vR, eta * vL, eta * vR))
    F = F + eta * np.where(b <= k, vL, vR)
