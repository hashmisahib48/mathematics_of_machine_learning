# Hand checks for Problems 34A.1 - 34A.4 (every number quoted in the solutions)
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
warnings.simplefilter('ignore', RuntimeWarning); warnings.simplefilter('ignore', ConvergenceWarning)
np.set_printoptions(precision=4, suppress=True)

print("== 34A.1  nodes A, u, v, B; links A-u 3, u-v 2, v-B 4, u-B 1")
W = np.zeros((4, 4))
for i, j, w in [(0, 1, 3), (1, 2, 2), (2, 3, 4), (1, 3, 1)]: W[i, j] = W[j, i] = w
d = W.sum(1); print("degrees A, u, v, B:", d)
U, L = [1, 2], [0, 3]; fL = np.array([0., 1.])
M = np.diag(d[U]) - W[np.ix_(U, U)]; rhs = W[np.ix_(U, L)] @ fL
print("matrix D_UU - W_UU:\n", M, "\nright-hand side W_UL f_L:", rhs, " det =", np.linalg.det(M))
fU = np.linalg.solve(M, rhs); print("f_u, f_v =", fU, " as 16 f:", 16 * fU)
f = np.array([0, fU[0], fU[1], 1.])
print("weighted means of the neighbours:", (W @ f / d)[U], " check 7/16, 13/16 =", 7 / 16, 13 / 16)
by_elim = (2 * (13 / 16) + 1) / 6; print("f_u from first equation with f_v = 13/16:", by_elim)

print("\n== 34A.2  chain 1 -2 -3, weights 1 and 3, alpha 0.5")
W = np.array([[0, 1, 0], [1, 0, 3], [0, 3, 0.]]); d = W.sum(1); al = 0.5
S = W / np.sqrt(np.outer(d, d)); print("degrees:", d, " S12 =", S[0, 1], " S23 =", S[1, 2], " S23^2 =", S[1, 2] ** 2)
Y = np.array([[1, 0], [0, 0], [0, 1.]]); F1 = al * S @ Y + (1 - al) * Y
print("alpha S Y:\n", al * S @ Y, "\n(1-alpha) Y:\n", (1 - al) * Y, "\nF after one step:\n", F1, " class of node 2:", F1[1].argmax())
# limit: (I - alpha S) F = (1 - alpha) Y, column by column
A = np.eye(3) - al * S; print("I - alpha S:\n", A)
Fl = (1 - al) * np.linalg.solve(A, Y); print("F limit:\n", Fl)
print("column 0 by hand: f2 =", 0.125 / (1 - al ** 2 * S[0, 1] ** 2 - al ** 2 * S[1, 2] ** 2), " (1/6 =", 1 / 6, ")")
print("column 1 by hand: f2 =", al * S[1, 2] * 0.5 / (1 - al ** 2 * S[0, 1] ** 2 - al ** 2 * S[1, 2] ** 2))
print("node 2 row:", Fl[1], " normalised:", Fl[1] / Fl[1].sum())
ls = LabelSpreading(kernel=lambda a, b: W.copy(), alpha=al, max_iter=100000, tol=1e-14).fit(np.arange(3.).reshape(-1, 1), np.array([0, -1, 1]))
print("scikit-learn label_distributions_[1] =", ls.label_distributions_[1], " max diff all rows:", abs(ls.label_distributions_ - Fl / Fl.sum(1, keepdims=True)).max())

print("\n== 34A.3  links 1-2 (2), 2-3 (1), 4-5 (3); nodes 1 (class 0) and 3 (class 1) labelled")
W = np.zeros((5, 5))
for i, j, w in [(0, 1, 2), (1, 2, 1), (3, 4, 3)]: W[i, j] = W[j, i] = w
d = W.sum(1); U = [1, 3, 4]; print("degrees:", d)
M = np.diag(d[U]) - W[np.ix_(U, U)]; print("D_UU - W_UU for nodes 2, 4, 5:\n", M, "\ndet =", np.linalg.det(M), " rank =", np.linalg.matrix_rank(M))
print("null vector (0, 1, 1): M @ v =", M @ np.array([0, 1, 1.]), " block 4,5 det =", np.linalg.det(M[1:, 1:]))
print("f_2 = (2*0 + 1*1)/3 =", (2 * 0 + 1 * 1) / 3)
lp = LabelPropagation(kernel=lambda a, b: W.copy(), max_iter=100000, tol=1e-12).fit(np.arange(5.).reshape(-1, 1), np.array([0, -1, 1, -1, -1]))
print("label_distributions_:\n", lp.label_distributions_, "\ntransduction_:", lp.transduction_, " classes_:", lp.classes_, " n_iter_:", lp.n_iter_)

print("\n== 34A.4  samples at 0, 1, 3, 7; knn with 2 neighbours, itself included")
x = np.array([0., 1, 3, 7]).reshape(-1, 1); y = np.array([0, -1, -1, 1])
from sklearn.neighbors import NearestNeighbors
Wk = NearestNeighbors(n_neighbors=2).fit(x).kneighbors_graph(x, 2, mode='connectivity').toarray()
print("W:\n", Wk.astype(int), "\ncolumn sums:", Wk.sum(0), " row sums:", Wk.sum(1))
T = Wk / Wk.sum(1, keepdims=True); print("T:\n", T)
f = np.array([0., 0.5, 0.5, 1.])                   # share of class 1, start at 1/2
print("iteration (share of class 1 at samples 2, 3):")
for t in range(0, 7):
    print(f"  t = {t}:", f[1:3])
    f = T @ f; f[[0, 3]] = [0, 1]
lp = LabelPropagation(kernel='knn', n_neighbors=2, max_iter=100000, tol=1e-12).fit(x, y)
print("scikit-learn transduction_:", lp.transduction_, "\nlabel_distributions_:\n", lp.label_distributions_)

print("\n== slips mentioned in the instructor notes")
W1 = np.zeros((4, 4))
for i, j, w in [(0, 1, 3), (1, 2, 2), (2, 3, 4), (1, 3, 1)]: W1[i, j] = W1[j, i] = w
wrong = np.linalg.solve(np.array([[5., -2], [-2, 6]]), np.array([1., 4]))
print("34A.1 with d_u = 5 (link u-B left out of the degree):", wrong)
T3 = np.array([[0, 1, 0], [0.25, 0, 0.75], [0, 1, 0.]])
Y3 = np.array([[1, 0], [0, 0], [0, 1.]])
print("34A.2 with T = D^-1 W instead of S, one step:", (0.5 * T3 @ Y3 + 0.5 * Y3)[1])
