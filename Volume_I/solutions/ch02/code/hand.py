# Hand checks for Problems 2A.1 - 2A.4 (every number quoted in the solutions)
import numpy as np
np.set_printoptions(precision=4, suppress=True)

print("== 2A.1  x = (2, -3, 6), z = (1, -3, 2)")
x = np.array([2., -3, 6]); z = np.array([1., -3, 2])
print("squares:", x ** 2, " sum =", (x ** 2).sum(), " L2 =", np.sqrt((x ** 2).sum()))
print("absolute values:", np.abs(x), " L1 =", np.abs(x).sum(), " Linf =", np.abs(x).max())
print("unit vector x/7:", x / 7, " its length:", np.linalg.norm(x / 7))
d = x - z; print("x - z =", d, " squares:", d ** 2, " sum =", (d ** 2).sum(), " distance =", np.sqrt((d ** 2).sum()))
print("norms agree with numpy:", [float(np.linalg.norm(x, q)) for q in (2, 1, np.inf)])

print("\n== 2A.2  b = (4, 3) onto a = (1, 2)")
a = np.array([1., 2]); b = np.array([4., 3])
print("a.b =", a @ b, " a.a =", a @ a, " c =", (a @ b) / (a @ a))
c = (a @ b) / (a @ a); p = c * a; r = b - p
print("p =", p, " r =", r, " a.r =", a @ r, " p.r =", p @ r)
print("Pythagoras: |b|^2 =", b @ b, " |p|^2 + |r|^2 =", p @ p + r @ r, "(", p @ p, "+", r @ r, ")")
print("any other multiple is worse: c = 1.5 gives |b - c a|^2 =", ((b - 1.5 * a) ** 2).sum(), " c = 2 gives", (r ** 2).sum())

print("\n== 2A.3  M = [[3,1],[5,2]]")
M = np.array([[3., 1], [5, 2]]); det = 3 * 2 - 1 * 5
print("det = 3*2 - 1*5 =", det)
Mi = np.array([[2., -1], [-5, 3]]) / det; print("inverse by Eq. 2.7-3:\n", Mi, "\nM @ inverse =\n", M @ Mi)
bv = np.array([1., 2]); xs = Mi @ bv; print("x = inverse @ (1, 2) =", xs, " check M x =", M @ xs, " numpy solve:", np.linalg.solve(M, bv))
print("image of the unit square has corners:", (M @ np.array([[0, 1, 1, 0], [0, 0, 1, 1]])).T.tolist(), " area =", abs(det))

print("\n== 2A.4  S = [[3,1],[1,3]]")
S = np.array([[3., 1], [1, 3]]); tr = np.trace(S); dt = np.linalg.det(S)
print("trace =", tr, " det =", round(dt, 6), " lambda^2 - 6 lambda + 8 roots:", np.roots([1, -tr, dt]))
for lam in (4., 2.): print(f"S - {lam:g} I =", (S - lam * np.eye(2)).tolist())
q1 = np.array([1., 1]) / np.sqrt(2); q2 = np.array([1., -1]) / np.sqrt(2)
print("S q1 =", S @ q1, " 4 q1 =", 4 * q1, "| S q2 =", S @ q2, " 2 q2 =", 2 * q2, "| q1.q2 =", q1 @ q2)
P1 = np.outer(q1, q1); P2 = np.outer(q2, q2)
print("q1 q1^T =", P1.tolist(), " q2 q2^T =", P2.tolist())
print("4 q1q1^T =", (4 * P1).tolist(), " 2 q2q2^T =", (2 * P2).tolist(), " sum =", (4 * P1 + 2 * P2).tolist())
print("sum of eigenvalues =", 4 + 2, " product =", 4 * 2)
print("numpy eigh:", np.linalg.eigh(S)[0])
print("\ncommon slips: 2A.1 forgetting the root, L2 squared =", (x ** 2).sum(),
      "; 2A.2 using c = a.b = ", a @ b, " (without dividing by a.a) gives p =", (a @ b) * a)
