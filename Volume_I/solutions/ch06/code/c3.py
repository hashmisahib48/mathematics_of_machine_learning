# Problem 6C.3: what doubling n does. Output: verify/c3.out
import numpy as np
s2, p = 0.0625, 4
print("  n   train    test     gap    test RMSE")
for n in (40, 80, 160, 320, 10**6):
    tr, te = s2 * (1 - p / n), s2 * (1 + p / n)
    print(f"{n:7d} {tr:.5f} {te:.5f} {te - tr:.5f} {np.sqrt(te):.4f}")
t40, t80 = s2 * (1 + p / 40), s2 * (1 + p / 80)
print(f"test error n=40 -> 80: {t40:.5f} -> {t80:.5f}, change {100 * (t80 / t40 - 1):.1f} per cent; floor {s2} would be {100 * (s2 / t40 - 1):.1f} per cent")
print(f"gap n=40 -> 80: {2*p*s2/40:.5f} -> {2*p*s2/80:.5f}")
print(f"halving the test error needs {t40 / 2:.5f} < sigma^2: impossible, since sigma^2 = {s2}; it would need sigma = {np.sqrt(t40 / 2 / (1 + p / 40)):.4f} at n = 40")
print(f"n for the test error to be within 1 per cent of sigma^2: {p / 0.01:.0f}")
