"""Numbers printed in Sections 3.2 and 3.4, computed step by step."""
import sys; sys.path.insert(0, '../code')
from linreg import LinearRegression

def mean(v): return sum(v) / len(v)

# ---- scores data, one feature (Figure 3.1-1, Eq. 3.2-7)
h = [1, 2, 3, 4, 5]; s = [52, 57, 61, 68, 72]
hb, sb = mean(h), mean(s)
Sxy = sum((a - hb) * (b - sb) for a, b in zip(h, s)); Sxx = sum((a - hb) ** 2 for a in h)
print('scores: xbar', hb, 'ybar', sb, 'Sxy', Sxy, 'Sxx', Sxx, 'w', Sxy / Sxx, 'b', round(sb - Sxy / Sxx * hb, 4))

# ---- Example 3.4-1, five deliveries
X = [[2, 1], [4, 2], [5, 1], [7, 3], [9, 2]]; y = [14, 23, 24, 35, 38]
m = LinearRegression().fit(X, y)
w1, w2 = m.coef_; b = m.intercept_
e = [yi - p for yi, p in zip(y, m.predict(X))]
x1b, x2b = mean([r[0] for r in X]), mean([r[1] for r in X]); yb = mean(y)
print('ex1: w', round(w1, 4), round(w2, 4), 'b', round(b, 4), 'pred', round(m.predict([[6, 2]])[0], 4))
print('ex1: residuals', [round(v, 3) for v in e])
print('ex1: sum e', round(sum(e), 10), ' sum x1~ e', round(sum((r[0] - x1b) * v for r, v in zip(X, e)), 10),
      ' sum x2~ e', round(sum((r[1] - x2b) * v for r, v in zip(X, e)), 10))
sst = sum((v - yb) ** 2 for v in y); sse = sum(v * v for v in e)
print('ex1: SStot', round(sst, 4), 'SSres', round(sse, 4), 'SSreg', round(sst - sse, 4), 'R2', round(1 - sse / sst, 4))

# ---- Example 3.4-2, fuel of a van, elimination with row swaps
L = [0.5, 1.0, 1.5, 1.0, 1.0]; D = [10, 20, 30, 40, 25]; S = [2, 3, 2, 4, 4]; F = [4.0, 6.5, 8.5, 11.0, 8.5]
X2 = [list(t) for t in zip(L, D, S)]
xb = [mean(L), mean(D), mean(S)]; fb = mean(F)
Xc = [[r[j] - xb[j] for j in range(3)] for r in X2]; fc = [v - fb for v in F]
A = [[sum(r[j] * r[k] for r in Xc) for k in range(3)] for j in range(3)]
c = [sum(r[j] * v for r, v in zip(Xc, fc)) for j in range(3)]
print('ex2: means', xb, round(fb, 4)); print('ex2: A', A, 'c', c)
A = [row[:] + [ci] for row, ci in zip(A, c)]           # augmented rows
for k in range(3):
    r = max(range(k, 3), key=lambda i: abs(A[i][k]))
    if r != k: A[k], A[r] = A[r], A[k]; print(f'ex2: k={k+1} swap rows {k+1} and {r+1}')
    for i in range(k + 1, 3):
        mm = A[i][k] / A[k][k]
        A[i] = [a - mm * q for a, q in zip(A[i], A[k])]
        print(f'ex2: k={k+1} m{i+1}{k+1} = {round(mm, 4)} row{i+1} =', [round(v, 4) for v in A[i]])
    print(f'ex2: after k={k+1}', [[round(v, 4) for v in row] for row in A])
w = [0.0] * 3
for k in (2, 1, 0):
    w[k] = (A[k][3] - sum(A[k][j] * w[j] for j in range(k + 1, 3))) / A[k][k]
print('ex2: w', [round(v, 4) for v in w], 'b', round(fb - sum(a * q for a, q in zip(xb, w)), 4))
m2 = LinearRegression().fit(X2, F)
print('ex2: class gives', [round(v, 4) for v in m2.coef_], round(m2.intercept_, 4), 'R2', round(m2.score(X2, F), 4))
print('ex2: predict 1.2 t, 35 km, 3 stops', round(m2.predict([[1.2, 35, 3]])[0], 4))

# ---- Section 3.1 and 3.2: the scores, three candidate lines and the step-by-step formulae
for (wl, bl) in [(6, 44), (5, 47), (5.1, 46.7)]:
    pr = [wl * a + bl for a in h]; er = [b_ - p for b_, p in zip(s, pr)]
    print('line', wl, bl, 'pred', [round(v, 1) for v in pr], 'err', [round(v, 1) for v in er], 'J', round(sum(v * v for v in er), 4))
xt = [a - hb for a in h]; yt = [b_ - sb for b_ in s]
print('xtilde', xt, 'ytilde', yt, 'xy', [a * b_ for a, b_ in zip(xt, yt)], 'xx', [a * a for a in xt])
print('Syy', sum(v * v for v in yt), 'R2', round(1 - 1.9 / sum(v * v for v in yt), 4))
print('pred 3.5 h', round(5.1 * 3.5 + 46.7, 2))
