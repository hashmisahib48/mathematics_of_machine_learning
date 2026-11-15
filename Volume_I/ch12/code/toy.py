# The calibration data, posterior and prediction by hand.
import numpy as np

x = np.array([-3.0, -1.0, 1.0, 3.0])   # setting (centred)
y = np.array([-2.0, -2.0, 1.0, 3.0])   # reading (centred)
alpha, lam = 1.0, 5.0                  # noise, weight precision

Sxx, Sxy = x @ x, x @ y
prec = lam + alpha * Sxx               # posterior precision
m = alpha * Sxy / prec                 # posterior mean
print(f"Sxx = {Sxx:.0f}   Sxy = {Sxy:.0f}   OLS w = {Sxy/Sxx:.3f}")
print(f"precision = {prec:.0f}   mean = {m:.3f}   "
      f"var = {1/prec:.3f}   sd = {prec**-0.5:.3f}")

print("prediction at x*:   mean    sd")      # Eq. 12.4-2
for xs in (0.0, 2.0, 6.0):
    sd = np.sqrt(1 / alpha + xs**2 / prec)
    print(f"  x* = {xs:3.0f}         {m*xs:6.3f}  {sd:.3f}")

for k in range(1, 5):                  # Eq. 12.4-8, repeated
    gamma = alpha * Sxx / (lam + alpha * Sxx)
    sse = np.sum((y - m * x) ** 2)
    lam = gamma / m**2
    alpha = (len(x) - gamma) / sse
    m = alpha * Sxy / (lam + alpha * Sxx)
    print(f"step {k}: gamma={gamma:.3f} sse={sse:.3f} "
          f"alpha={alpha:.3f} lambda={lam:.3f} w={m:.4f}")
