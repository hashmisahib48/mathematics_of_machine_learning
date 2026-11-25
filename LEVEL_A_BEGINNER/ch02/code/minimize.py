# Minimizing J(w) = sum (y_i - w x_i)^2 by a formula and by descent.
x = [1.0, 2.0, 3.0]
y = [2.0, 4.0, 7.0]

def J(w):
    return sum((b - w * a) ** 2 for a, b in zip(x, y))

def dJ(w):                               # derivative of J
    return -2 * sum(a * (b - w * a) for a, b in zip(x, y))

w_best = sum(a * b for a, b in zip(x, y)) / sum(a * a for a in x)
print("formula: w =", round(w_best, 4), " J =", round(J(w_best), 4))
h = 1e-6                                 # check dJ by a difference
num = (J(1 + h) - J(1)) / h
print("dJ(1) =", dJ(1.0), " numerical:", round(num, 3))
w = 0.0
for k in range(6):                       # steps against the slope
    w = w - 0.03 * dJ(w)
    print(f"step {k + 1}: w = {w:.4f}  J = {J(w):.4f}")
