# Numbers for Problem 4C.3 (the weight on the data)
import numpy as np
print("w = n/(n + alpha), the weight on the sample mean (Example 4.9-1: alpha = sigma^2/tau^2 = 4)")
ns = (1, 4, 10, 20, 36, 100, 1000)
print("alpha \\ n " + "".join(f"{n:8d}" for n in ns))
for a in (0.4, 4, 16, 100):
    print(f"{a:9.1f} " + "".join(f"{n / (n + a):8.3f}" for n in ns))
for a in (4, 16, 100):
    print(f"alpha = {a}: w = 0.5 at n = {a}; w = 0.9 at n = {9 * a}; shrinkage 1 - w = alpha/(n+alpha) is {a / (100 + a):.3f} at n = 100")
x = np.array([2., 3, 3, 4]); xbar = x.mean()
print("Example 4.9-1: n = 4, xbar =", xbar, " alpha = 4: w =", 4 / 8, " posterior mean =", 0.5 * xbar + 0.5 * 0)
