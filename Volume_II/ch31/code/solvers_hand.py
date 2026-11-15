# Three updates of one weight: by the formulas and by the library.
import numpy as np
from sklearn.neural_network._stochastic_optimizers import (
    AdamOptimizer, SGDOptimizer)

g = [2.0, 1.0, -1.0]                  # the gradient at three steps
eta, mu = 0.1, 0.9

def library(opt):                     # the library's own steps
    w = [np.array([1.0])]
    out = []
    for gt in g:
        opt.update_params(w, [np.array([gt])])
        out.append(w[0][0])
    return out

plain = [1.0 - eta * sum(g[:t + 1]) for t in range(3)]   # mu = 0
v, heavy, nest, w1, w2 = 0.0, [], [], 1.0, 1.0
for gt in g:
    v = mu * v - eta * gt                 # Eq. 31.5-2
    w1 += v; heavy.append(w1)             # classical momentum
    w2 += mu * v - eta * gt; nest.append(w2)   # Nesterov, as coded
m = s = 0.0; w3 = 1.0; adam = []
for t, gt in enumerate(g, start=1):   # Eq. 31.5-4
    m = 0.9 * m + 0.1 * gt
    s = 0.999 * s + 0.001 * gt**2
    step = eta * np.sqrt(1 - 0.999**t) / (1 - 0.9**t)
    w3 -= step * m / (np.sqrt(s) + 1e-8); adam.append(w3)

print("step      g   plain  momentum  Nesterov    Adam")
for t in range(3):
    print(f"{t + 1:3d}  {g[t]:6.1f}  {plain[t]:6.4f}  "
          f"{heavy[t]:7.4f}  {nest[t]:7.4f}  {adam[t]:6.4f}")
z = [np.zeros(1)]
ok = [np.allclose(library(SGDOptimizer(z, eta, "constant", mu, n)),
                  r) for n, r in ((False, heavy), (True, nest))]
ok.append(np.allclose(library(AdamOptimizer(z, eta)), adam))
print("agree with scikit-learn:", ok)
