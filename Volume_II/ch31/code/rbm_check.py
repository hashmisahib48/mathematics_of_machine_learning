import numpy as np
from sklearn.neural_network import BernoulliRBM

sig = lambda z: 1 / (1 + np.exp(-z))
V = np.array([[1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 0, 0]], float)

# One update of persistent contrastive divergence, Eq. 31.11-6
rbm = BernoulliRBM(n_components=2, learning_rate=0.1, batch_size=4,
                   random_state=0)
rbm.partial_fit(V)
rng = np.random.RandomState(0)
W = rng.normal(0, 0.01, (2, 3))               # b = c = 0 at first
h_pos = sig(V @ W.T)                          # P(h = 1 | data)
v_neg = rng.uniform(size=(4, 3)) < sig(np.zeros((4, 2)) @ W)
h_neg = sig(v_neg @ W.T)                      # P(h = 1 | fantasy)
W1 = W + 0.1 / 4 * (h_pos.T @ V - h_neg.T @ v_neg)
same = np.allclose(W1, rbm.components_)
print("update equals Eq. 31.11-6:", same)

# Free energy and conditionals against sums over all states
rbm.components_ = W = np.array([[1.0, -2.0, 0.5], [0.5, 1.0, -1.0]])
rbm.intercept_visible_ = b = np.array([0.1, -0.2, 0.3])
rbm.intercept_hidden_ = c = np.array([-0.5, 0.2])
Hs = np.array(list(np.ndindex(2, 2)), float)
Vs = np.array(list(np.ndindex(2, 2, 2)), float)
def energy(v, h):                             # Eq. 31.11-1
    return -b @ v - c @ h - h @ W @ v
weight = lambda v: sum(np.exp(-energy(v, h)) for h in Hs)
Z = sum(weight(v) for v in Vs)                # Eq. 31.11-2
F = -np.log([weight(v) for v in V])
print("F(v) =", np.round(F, 4), np.allclose(F, rbm._free_energy(V)))
print("P(v) =", np.round(np.exp(-F) / Z, 4), " all 8 sum to",
      round(sum(weight(v) for v in Vs) / Z, 6))
print("P(h=1|v) equals transform:",
      np.allclose(rbm.transform(V), sig(V @ W.T + c)))
