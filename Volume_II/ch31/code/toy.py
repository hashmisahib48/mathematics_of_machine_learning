# The 2-2-1 network of Chapter 31: one forward and backward pass.
import numpy as np

x = np.array([2.0, -1.0]); y = 1.0        # one XOR sample
W1 = np.array([[0.5, 1.0], [-1.0, 0.5]])  # coefs_[0], shape (2, 2)
b1 = np.array([0.5, -0.5])                # intercepts_[0]
W2 = np.array([[1.0], [-1.0]])            # coefs_[1], shape (2, 1)
b2 = np.array([0.0])                      # intercepts_[1]
eta = 0.5

def forward(W1, b1, W2, b2):
    z1 = x @ W1 + b1                      # Eq. 31.3-2
    h = np.maximum(z1, 0)                 # ReLU
    z2 = h @ W2 + b2
    p = 1 / (1 + np.exp(-z2))             # logistic output
    return z1, h, z2, p

z1, h, z2, p = forward(W1, b1, W2, b2)
loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
print("z1", z1, " h", h, " z2", z2, " p", np.round(p, 4))
print(f"loss {loss[0]:.4f}")
d2 = p - y                                # Eq. 31.4-1
d1 = (W2 @ d2) * (z1 > 0)                 # Eq. 31.4-3
gW2 = np.outer(h, d2); gb2 = d2           # Eq. 31.4-4
gW1 = np.outer(x, d1); gb1 = d1
print("delta2", np.round(d2, 4), " delta1", np.round(d1, 4))
print("gW2", np.round(gW2.ravel(), 4), " gb2", np.round(gb2, 4))
print("gW1", np.round(gW1, 4).tolist(), " gb1", np.round(gb1, 4))

def J(v):                                 # loss of all 9 numbers
    W1_, b1_ = v[:4].reshape(2, 2), v[4:6]
    W2_, b2_ = v[6:8, None], v[8:]
    pp = forward(W1_, b1_, W2_, b2_)[3]
    return -(y * np.log(pp) + (1 - y) * np.log(1 - pp))[0]

v = np.r_[W1.ravel(), b1, W2.ravel(), b2]
g = np.r_[gW1.ravel(), gb1, gW2.ravel(), gb2]
num = np.array([(J(v + 1e-6 * e) - J(v - 1e-6 * e)) / 2e-6
                for e in np.eye(9)])      # Eq. 3.10-1 (Volume I)
print("largest gradient error", f"{abs(num - g).max():.1e}")

v2 = v - eta * g                          # one gradient step
print("new W1", np.round(v2[:4], 4), "b1", np.round(v2[4:6], 4))
print("new W2", np.round(v2[6:8], 4), "b2", np.round(v2[8:], 4))
print(f"loss after the step {J(v2):.4f}")
