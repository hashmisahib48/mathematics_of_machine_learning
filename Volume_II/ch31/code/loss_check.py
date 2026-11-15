# Facts about the source of release 1.9.1 that the text states (not a listing).
import contextlib, io, warnings
import numpy as np
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neural_network._multilayer_perceptron import _pack
from sklearn.utils import check_random_state

warnings.simplefilter("ignore")
rng = np.random.RandomState(0)
X = rng.randn(40, 3); n = 40
y2 = (X[:, 0] > 0).astype(int); y3 = rng.randint(0, 3, n)
yr = X[:, 0] + 0.1 * rng.randn(n)
alpha = 0.3

def out(m, A):                              # linear scores of the last layer
    for i, (W, b) in enumerate(zip(m.coefs_, m.intercepts_)):
        A = A @ W + b
        A = np.maximum(A, 0) if i < len(m.coefs_) - 1 else A
    return A
def pen(m, size):                           # alpha/(2 size) sum ||W||^2
    return alpha / (2 * size) * sum((W**2).sum() for W in m.coefs_)
def softmax_loss(Z, y):
    Z = Z - Z.max(axis=1, keepdims=True)
    return -(Z[np.arange(len(y)), y] - np.log(np.exp(Z).sum(axis=1))).mean()

print("1. loss_ and Eqs. 31.3-6 to 31.3-8 (L-BFGS, full batch)")
kw = dict(solver="lbfgs", alpha=alpha, max_iter=200, random_state=0)
m = MLPClassifier((6,), **kw).fit(X, y2); z = out(m, X)[:, 0]
f = np.mean(np.logaddexp(0, z) - y2 * z) + pen(m, n)
print(f"   binary      {m.loss_:.10f} {f:.10f}  {m.out_activation_}")
m = MLPClassifier((6,), **kw).fit(X, y3)
print(f"   3 classes   {m.loss_:.10f} "
      f"{softmax_loss(out(m, X), y3) + pen(m, n):.10f}  {m.out_activation_}")
m = MLPRegressor(hidden_layer_sizes=(6,), **kw).fit(X, yr)
f = 0.5 * np.mean((yr - out(m, X)[:, 0])**2) + pen(m, n)
print(f"   regression  {m.loss_:.10f} {f:.10f}  {m.out_activation_}")

print("2. penalty in a mini-batch of 10 of the 40 rows")
m = MLPClassifier((6,), solver="sgd", alpha=alpha, max_iter=1,
                  batch_size=10, random_state=0).fit(X, y3)
sizes = [X.shape[1], 6, 3]
acts = [X[:10]] + [None] * 2; dl = [None] * 2
cg = [np.empty((a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
ig = [np.empty(b) for b in sizes[1:]]
Yb = m._label_binarizer.transform(y3[:10]).astype(bool)
J = m._backprop(X[:10], Yb, None, acts, dl, cg, ig)[0]
data = softmax_loss(out(m, X[:10]), y3[:10])
print(f"   _backprop {J:.6f}")
print(f"   data + penalty/10 {data + pen(m, 10):.6f}")
print(f"   data + penalty/40 {data + pen(m, 40):.6f}")

print("3. initial weights lie in +-sqrt(f/(fan_in + fan_out))")
c = MLPClassifier(); c._random_state = check_random_state(0)
for act, fac in (("relu", 6), ("logistic", 2)):
    c.activation = act
    W, b = c._init_coef(200, 300, np.float64)
    bound = np.sqrt(fac / 500)
    print(f"   {act:8s} bound {bound:.5f}  max|w| {abs(W).max():.5f}"
          f"  max|b| {abs(b).max():.5f}")

print("4. two targets: gradient / gradient of reported loss")
Y2 = np.c_[yr, 2 * yr + X[:, 1]]
m = MLPRegressor(hidden_layer_sizes=(5,), solver="lbfgs", max_iter=1,
                 random_state=0).fit(X, Y2)
m._coef_indptr = [(0, 15, (3, 5)), (15, 25, (5, 2))]
m._intercept_indptr = [(25, 30), (30, 32)]
sizes = [3, 5, 2]
acts = [X] + [None] * 2; dl = [None] * 2
cg = [np.empty((a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
ig = [np.empty(b) for b in sizes[1:]]
def lg(v): return m._loss_grad_lbfgs(v, X, Y2, None, acts, dl, cg, ig)
v = _pack(m.coefs_, m.intercepts_)
g = lg(v)[1]
num = np.array([(lg(v + 1e-6 * e)[0] - lg(v - 1e-6 * e)[0]) / 2e-6
                for e in np.eye(len(v))])
big = abs(num) > 1e-3
print(f"   median ratio over {big.sum()} entries: "
      f"{np.median(g[big] / num[big]):.4f}")

print("5. learning rate 'adaptive' (n_iter_no_change=3, tol=1000)")
X2, y = make_moons(100, noise=0.3, random_state=0)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    MLPClassifier((5,), solver="sgd", learning_rate="adaptive",
                  learning_rate_init=0.1, tol=1000, n_iter_no_change=3,
                  max_iter=12, random_state=0, verbose=True).fit(X2, y)
epoch, first = 0, []
for line in buf.getvalue().splitlines():
    if line.startswith("Iteration"):
        epoch = int(line.split()[1].strip(","))
    if "Setting learning rate" in line:
        first.append((epoch, line.split()[-1]))
print("   cut by 5 after epochs:", first)

print("6. 'invscaling' after each epoch: eta_0 / (t + 1)^power_t")
m = MLPClassifier((5,), solver="sgd", learning_rate="invscaling",
                  learning_rate_init=0.1, power_t=0.5, batch_size=100,
                  random_state=0)
for e in range(3):
    m.partial_fit(X2, y, classes=[0, 1])
    print(f"   t_ = {m.t_}: {m._optimizer.learning_rate:.6f}"
          f"  formula {0.1 / (m.t_ + 1)**0.5:.6f}")

print("7. early stopping (n_iter_no_change=5)")
m = MLPClassifier((50,), early_stopping=True, n_iter_no_change=5,
                  validation_fraction=0.2, random_state=0,
                  max_iter=500).fit(X2, y)
best = int(np.argmax(m.validation_scores_)) + 1
print(f"   n_iter_ {m.n_iter_}, best epoch {best}, t_ {m.t_},"
      f" best_loss_ {m.best_loss_}")
