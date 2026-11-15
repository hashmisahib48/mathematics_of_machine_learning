# Hand checks for Problems 43A.1 - 43A.3 and 43C.1 (every number quoted in the solutions)
import numpy as np
np.set_printoptions(precision=4, suppress=True)

print("== 43A.1  one KL update, X = [[3,6],[6,3]], W = [[1,2],[2,1]], H0 = ones")
X = np.array([[3, 6], [6, 3.]]); W = np.array([[1, 2], [2, 1.]]); H0 = np.ones((2, 2))
kl = lambda X, Y: X * np.log(X / Y) - X + Y                 # d_1(x|y) entry by entry
Y0 = W @ H0; print("WH0 =\n", Y0)
R = X / Y0; print("X/WH0 =\n", R)
num = W.T @ R; den = W.sum(axis=0)[:, None] * np.ones((1, 2))
print("W^T (X/WH0) =\n", num, "\ncolumn sums of W (= W^T 1):", W.sum(axis=0))
H1 = H0 * num / den; print("H1 =\n", H1, "\n  as fractions x3:", 3 * H1)
Y1 = W @ H1; print("WH1 =\n", Y1)
print("entries of d_1 before:\n", kl(X, Y0), "\n sum D0 =", kl(X, Y0).sum())
print("entries of d_1 after:\n", kl(X, Y1), "\n sum D1 =", kl(X, Y1).sum())
print("6 ln2 - 3 =", 6*np.log(2) - 3, " 3 ln(6/4.3333)-3+4.3333 =", 3*np.log(3/Y1[0,0]) - 3 + Y1[0,0], " 6 ln(6/4.6667) - 6 + 4.6667 =", 6*np.log(6/Y1[0,1]) - 6 + Y1[0,1])
print("Frobenius error before / after:", np.linalg.norm(X - Y0), np.linalg.norm(X - Y1))
print("sum of X, sum of WH1 (KL updates keep the total of the fit equal to that of the data?):", X.sum(), Y1.sum())
H = H1.copy()
for it in range(200):
    H = H * (W.T @ (X / (W @ H))) / W.sum(axis=0)[:, None]
print("after 200 updates: D =", kl(X, W @ H).sum(), " H =\n", H)

print("\n== 43A.2  scale of the losses")
d2 = lambda x, y: 0.5 * (x - y) ** 2
d1 = lambda x, y: x * np.log(x / y) - x + y
d0 = lambda x, y: x / y - np.log(x / y) - 1
for x, y in ((4, 2), (40, 20)):
    print(f"x={x}, y={y}: d2 = {d2(x,y):.4f}   d1 = {d1(x,y):.4f}   d0 = {d0(x,y):.4f}   (4 ln 2 - 2 = {4*np.log(2)-2:.4f})")
print("ratios d2, d1, d0:", d2(40, 20) / d2(4, 2), d1(40, 20) / d1(4, 2), d0(40, 20) / d0(4, 2))
print("x=4,y=2 ; reversed x=2,y=4: d2", d2(4, 2), d2(2, 4), " d1", d1(4, 2), d1(2, 4))

print("\n== 43A.3  kurtosis of three samples")
smp = {'a': np.array([-1, -1, -1, 1, 1, 1.]), 'b': np.array([-2, -1, 0, 0, 1, 2.]), 'c': np.array([0]*8 + [-3, 3.])}
for k, v in smp.items():
    print(f"{k}: n={len(v)} mean={v.mean():.1f} sum y^2={np.sum(v**2):.0f} sum y^4={np.sum(v**4):.0f}  var={np.mean(v**2):.4f} m4={np.mean(v**4):.4f}"
          f"  m4/var^2={np.mean(v**4)/np.mean(v**2)**2:.4f}  kurtosis={np.mean(v**4)/np.mean(v**2)**2-3:.4f}")
print("a: m4/var^2 for a two-point symmetric variable = 1;  Gaussian = 3")
print("scaled to unit variance, c has |y| =", 3/np.sqrt(1.8), " fourth power", (3/np.sqrt(1.8))**4)

print("\n== 43C.1  other exact factorisations of the three paints")
Xp = np.array([[4, 2], [3, 3], [2, 4.]])
print("W = X, H = I:\n", Xp @ np.eye(2), " reproduces X:", np.allclose(Xp @ np.eye(2), Xp))
print("pigments (2,1), (1,2): W =", np.linalg.solve(np.array([[2, 1], [1, 2.]]).T, Xp.T).T.tolist())
ang = lambda v: np.degrees(np.arctan2(v[1], v[0]))
print("angles of the tins:", [round(ang(r), 2) for r in Xp], " pigments:", ang([2, 1]), ang([1, 2]))
print("limits: row 1 of H between", 0.0, "and", ang([2, 1]), "deg; row 2 between", ang([1, 2]), "and 90 deg")
rng = np.random.RandomState(1)
for _ in range(3):
    t1 = np.radians(rng.uniform(0, ang([2, 1]))); t2 = np.radians(rng.uniform(ang([1, 2]), 90))
    Hn = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]]); Wn = Xp @ np.linalg.inv(Hn)
    print(f"  angles {np.degrees(t1):5.1f}, {np.degrees(t2):5.1f}: min of W = {Wn.min():.3f} (>= 0), exact: {np.allclose(Wn @ Hn, Xp)}")
t1 = np.radians(30); Hn = np.array([[np.cos(t1), np.sin(t1)], [1, 2]]); Wn = Xp @ np.linalg.inv(Hn)
print("  a first ray at 30 deg (outside the limit 26.57): min of W =", Wn.min().round(3), "(negative)")
Wp = np.linalg.solve(np.array([[2, 1], [1, 2.]]).T, Xp.T).T; Hp = np.array([[2, 1], [1, 2.]])
D = np.diag([3.0, 0.5]); Pm = np.array([[0, 1], [1, 0.]])
W2 = Wp @ np.linalg.inv(D) @ Pm.T; H2 = Pm @ D @ Hp
print("scaled by D = diag(3, 0.5) and swapped: W' =\n", W2, "\nH' =\n", H2, "\n product equals X:", np.allclose(W2 @ H2, Xp), " W' >= 0:", (W2 >= 0).all(), " H' >= 0:", (H2 >= 0).all())
