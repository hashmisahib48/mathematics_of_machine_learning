# Question 1 of Chapter 56: a grid and random search at equal budget when p settings exist but only two matter
import math
import numpy as np

rng = np.random.RandomState(0)
side = 0.2                                   # good region: a box of side 0.2 in each of the two settings that matter
q = side ** 2
print("good region: fraction q = %.2f of the unit square of the two important settings" % q)
print("draws for P = 0.95 (Eq. 56.3-3), any number of settings: %d" % math.ceil(math.log(0.05) / math.log(1 - q)))
print("\n p  grid m^p   grid P(hit)  random P(hit, same n)   simulated random")
for p in (2, 3, 4, 5, 6):
    m = 2 if p >= 4 else 4                   # values per setting, chosen to keep the grid small
    n = m ** p
    vals = (np.arange(m) + 0.5) / m          # grid values of one setting (cell centres)
    ph = np.mean([np.any(np.abs(vals - c) <= side / 2) for c in rng.uniform(side / 2, 1 - side / 2, 20000)]) ** 2
    pr = 1 - (1 - q) ** n
    sim = np.mean([np.any(np.all(np.abs(rng.uniform(0, 1, (n, 2)) - rng.uniform(side / 2, 1 - side / 2, 2)) <= side / 2, axis=1)) for _ in range(4000)])
    print("%2d  %4d (m=%d)   %.3f        %.3f                  %.3f" % (p, n, m, ph, pr, sim))
print("\ngrid sizes 5^p:", {p: 5 ** p for p in (2, 3, 5, 7)}, " fits with K = 5:", {p: 5 * 5 ** p + 1 for p in (2, 3, 5, 7)})
