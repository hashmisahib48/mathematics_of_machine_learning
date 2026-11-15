# Numbers quoted in Examples R.3-1, R.4-1, R.5-1, R.7-1 and the text.
import json
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import KFold

N = {}
# R.3-1: J(w) = w1^2 + w1 w2 + w2^2 - 3 w1 - 3 w2
Hm = np.array([[2., 1], [1, 2]])
grad = lambda w: Hm @ w - 3
J = lambda w: w @ Hm @ w / 2 - 3 * w.sum()
w = np.zeros(2); gd = [w.copy()]
for _ in range(4):
    w = w - 0.5 * grad(w); gd.append(w.copy())
w = np.zeros(2); cd = [w.copy()]
for _ in range(3):
    w[0] = (3 - w[1]) / 2; cd.append(w.copy())
    w[1] = (3 - w[0]) / 2; cd.append(w.copy())
N['hess_eig'] = np.linalg.eigvalsh(Hm).tolist()
N['gd'] = [(a.tolist(), float(J(a))) for a in gd]
N['cd'] = [(a.tolist(), float(J(a))) for a in cd]
g0 = grad(np.zeros(2))
N['newton'] = (-np.linalg.solve(Hm, g0)).tolist()
N['J0'] = float(J(np.zeros(2))); N['Jmin'] = float(J(np.ones(2)))
# R.4-1 (c): log-density of a bivariate normal
mu = np.array([1., 2]); x = np.array([3., 3])
Sg = np.array([[4., 2], [2, 3]])
d = x - mu; m2 = float(d @ np.linalg.solve(Sg, d))
N['maha2'] = m2; N['logdet'] = float(np.log(np.linalg.det(Sg)))
N['logpdf'] = float(multivariate_normal(mu, Sg).logpdf(x))
N['logpdf_by_hand'] = float(
    -0.5 * (2 * np.log(2 * np.pi) + N['logdet'] + m2))
# R.5-1: three points
Pt = np.array([[0., 0], [3, 4], [1, 1]])
def dist(a, b, q):
    return float((np.abs(a - b) ** q).sum() ** (1 / q)) if q < 99 \
        else float(np.abs(a - b).max())
N['dist'] = {f"{i}{j}": [dist(Pt[i], Pt[j], q) for q in (1, 2, 100)]
             for i, j in ((0, 1), (0, 2), (1, 2))}
K = rbf_kernel(Pt, gamma=0.1)
N['K'] = K.round(4).tolist()
N['Keig'] = np.linalg.eigvalsh(K).tolist()
# R.7-1: four-fold cross-validation of the mean predictor
y = np.array([5., 11, 2, 8, 13, 4, 10, 7]); rows = []
for tr, te in KFold(4).split(y):
    m = y[tr].mean(); e = float(((y[te] - m) ** 2).mean())
    rows.append((tr.tolist(), te.tolist(), float(m), e))
N['cv'] = rows; N['cv_mean'] = float(np.mean([r[3] for r in rows]))
N['var'] = float(y.var())
# R.2-1: Rayleigh quotient of S = A'A
S = np.array([[5., 2], [2, 2]])
N['rayleigh'] = [float(v @ S @ v / (v @ v)) for v in
                 (np.array([1., 0]), np.array([0., 1]), np.array([2., 1]))]
# R.8: Laplacians of the two graphs, and normalised forms
def lap(E):
    M = np.zeros((5, 5))
    for i, j in E:
        M[i, j] = M[j, i] = 1
    return np.diag(M.sum(1)) - M
two = [(0, 1), (1, 2), (0, 2), (3, 4)]
N['L_two'] = (np.linalg.eigvalsh(lap(two)) + 0.0).round(4).tolist()
N['L_bridge'] = (np.linalg.eigvalsh(lap(two + [(2, 3)]))).round(4).tolist()
# R.8: normalised Laplacians of the bridge graph
W = np.zeros((5, 5))
for i, j in [(0, 1), (1, 2), (0, 2), (3, 4), (2, 3)]:
    W[i, j] = W[j, i] = 1
dg = W.sum(1); Ls = np.eye(5) - W / np.sqrt(np.outer(dg, dg))
Lr = np.eye(5) - W / dg[:, None]
N['Lsym'] = np.linalg.eigvalsh(Ls).tolist()
N['Lrw'] = np.sort(np.linalg.eigvals(Lr).real).tolist()
json.dump(N, open("verify/numbers.json", "w"), indent=1)
for k, v in N.items():
    print(k, v)
Lb = np.diag(W.sum(1)) - W
f = np.array([2., 2, 2, -3, -3])
print("f'Lf, f'f, ratio:", f @ Lb @ f, f @ f, round(f @ Lb @ f / (f @ f), 4))
