# Bayes' rule for a screening test, and distances between flowers.
import numpy as np

p_d = 0.01                               # P(disease)
p_pos_d, p_pos_h = 0.95, 0.10      # P(+ | disease), P(+ | healthy)
p_pos = p_pos_d * p_d + p_pos_h * (1 - p_d)
print("P(+) =", round(p_pos, 4))
print("P(disease | +) =", round(p_pos_d * p_d / p_pos, 4))

a = np.array([5.1, 3.5, 1.4, 0.2])       # setosa, iris 0
b = np.array([7.0, 3.2, 4.7, 1.4])       # versicolor, iris 50
c = np.array([6.3, 3.3, 6.0, 2.5])       # virginica, iris 100
for name, p, q in [("a-b", a, b), ("a-c", a, c), ("b-c", b, c)]:
    d2 = np.sqrt(((p - q) ** 2).sum())
    d1 = np.abs(p - q).sum()
    print(f"{name}: Euclidean {d2:.3f}  Manhattan {d1:.1f}")
