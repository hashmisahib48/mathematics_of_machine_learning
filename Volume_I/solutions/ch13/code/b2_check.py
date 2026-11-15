# Break-points of the penalised median fit: where does the solution leave the LAD line, and where does w reach 0?
import numpy as np
x = np.arange(1., 6); y = np.array([3., 4, 6, 7, 20])
def L(w):                                    # profile of the mean pinball loss (tau = 0.5) over the intercept
    r = y - w * x; b = np.median(r); return np.mean(0.5 * np.abs(r - b))
h = 1e-6
for w0 in (1.5, 1.0, 0.5, 0.0):
    print("w = %.2f: L = %.4f, slope of L to the left %.4f, to the right %.4f" % (w0, L(w0), (L(w0) - L(w0 - h)) / h, (L(w0 + h) - L(w0)) / h))
for wm in (1.45, 1.2, 0.5):
    print("slope of L at w = %.2f: %.4f" % (wm, (L(wm + h) - L(wm - h)) / (2 * h)))
print("L(w) for w = 1.5 ... 2.0:", [round(float(L(w)), 4) for w in (1.5, 1.6, 1.75, 1.9, 2.0, 2.1)])
# for a given alpha, minimise L(w) + alpha |w|
for a in (0.1, 0.19, 0.21, 0.3, 0.45, 0.5, 0.55, 0.59, 0.61, 1.0):
    ws = np.linspace(-1, 4, 50001); v = np.array([L(w) + a * abs(w) for w in ws]); wb = ws[v.argmin()]
    print("alpha %.2f: best slope on a grid %.3f, objective %.4f" % (a, wb, v.min()))

# points through which each candidate line passes
xs = x
for nm, (w, b) in {"(1.5, 1.5)": (1.5, 1.5), "(1.333, 1.667)": (4 / 3, 5 / 3), "slope 1.0 end of the flat segment": (1.0, None)}.items():
    if b is None: b = np.median(y - w * xs)
    print("line %s, intercept %.3f passes through points %s" % (nm, b, np.where(abs(y - b - w * xs) < 1e-9)[0] + 1))
