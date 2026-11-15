# Numbers quoted in the model answers to the questions of Chapter 31.
import warnings
import numpy as np
from sklearn.datasets import load_digits, load_diabetes, make_moons
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegressionCV, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM, MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
warnings.simplefilter("ignore", ConvergenceWarning); warnings.simplefilter("ignore", UserWarning)

print("== Q1  identity activations")
X8 = np.array([[-2, -1], [-1, -2], [1, 2], [2, 1], [-2, 1], [-1, 2], [1, -2], [2, -1]], float); y8 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
net = MLPClassifier((4, 4), activation="identity", solver="lbfgs", max_iter=2000, random_state=0).fit(X8, y8)
print("parts (two hidden layers of 4, identity): largest |p - 0.5| over the eight parts", float(np.abs(net.predict_proba(X8)[:, 1] - 0.5).max()).__round__(4), " loss", round(net.loss_, 4), " ln 2 =", round(np.log(2), 4))
Xd, yd = load_diabetes(return_X_y=True); Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.25, random_state=0)
sx = StandardScaler().fit(Xtr); A, B = sx.transform(Xtr), sx.transform(Xte); m, s = ytr.mean(), ytr.std()
reg = MLPRegressor(hidden_layer_sizes=(10, 10, 10), activation="identity", solver="lbfgs", alpha=0, max_iter=5000, tol=1e-12, random_state=0).fit(A, (ytr - m) / s)
W, b = np.eye(10), np.zeros(10)
for Wl, bl in zip(reg.coefs_, reg.intercepts_): W, b = W @ Wl, b @ Wl + bl     # collapse the layers one by one
print("four identity weight matrices collapse into one of shape", W.shape, "; B @ W + b agrees with predict to", f"{np.abs((B @ W + b)[:, 0] - reg.predict(B)).max():.1e}")
lin = LinearRegression().fit(A, (ytr - m) / s)
print(f"test R2 of that network {1 - ((yte - (reg.predict(B) * s + m)) ** 2).sum() / ((yte - yte.mean()) ** 2).sum():.4f}   least squares {lin.score(B, (yte - m) / s) if False else 1 - ((yte - (lin.predict(B) * s + m)) ** 2).sum() / ((yte - yte.mean()) ** 2).sum():.4f}")
print("largest difference between the collapsed weights and the least-squares coefficients:", f"{np.abs(W[:, 0] - lin.coef_).max():.1e}")

print("\n== Q2  symmetry: equal weights in the output layer, random hidden weights")
Xm, ym = make_moons(100, noise=0.3, random_state=0); Xm = StandardScaler().fit_transform(Xm)
def run(setup, k=2, steps=60):
    n = MLPClassifier((4,), solver="sgd", momentum=0, learning_rate_init=0.5, batch_size=100, max_iter=1, random_state=0, alpha=0).fit(Xm, ym)
    setup(n)
    for _ in range(steps): n.partial_fit(Xm, ym, classes=[0, 1])
    return n
def equal_out(n): n.coefs_[1][:] = 0.5; n.intercepts_[1][:] = 0.0
def equal_hidden(n):
    n.coefs_[0][:] = n.coefs_[0][:, [0]]; n.intercepts_[0][:] = n.intercepts_[0][0]; n.coefs_[1][:] = 0.5
n1 = run(equal_out); print("equal output weights, random hidden: spread of the 4 output weights after 60 updates", np.ptp(n1.coefs_[1]).round(4), " weights", n1.coefs_[1][:, 0].round(3).tolist())
n2 = run(equal_hidden); print("equal hidden units, equal output weights: spread of the hidden weight columns", np.ptp(n2.coefs_[0], axis=1).max(), " output weights", n2.coefs_[1][:, 0].round(3).tolist())
Xi = np.random.default_rng(0).normal(size=(120, 3)); yi = np.repeat([0, 1, 2], 40); Xi += np.eye(3)[yi] * 2
n3 = MLPClassifier((4,), solver="sgd", momentum=0, learning_rate_init=0.5, batch_size=120, max_iter=1, random_state=0, alpha=0).fit(Xi, yi)
n3.coefs_[1][:] = 0.3; n3.intercepts_[1][:] = 0.0
for _ in range(60): n3.partial_fit(Xi, yi, classes=[0, 1, 2])
print("three classes, the three output columns start equal: distance between columns 0 and 1 after 60 updates", np.linalg.norm(n3.coefs_[1][:, 0] - n3.coefs_[1][:, 1]).round(3))

print("\n== Q3  one feature in thousands (two moons, 300 samples, 16 hidden units)")
X, y = make_moons(300, noise=0.3, random_state=0); Xa, Xb, ya, yb = train_test_split(X, y, test_size=75, random_state=0)
big = np.array([1000.0, 1.0])
for nm, f in [("both features on the same scale", np.array([1.0, 1.0])), ("x1 in thousands (x 1000)", big)]:
    for raw in (True, False):
        P, Q = Xa * f, Xb * f
        if not raw: sc = StandardScaler().fit(P); P, Q = sc.transform(P), sc.transform(Q)
        r = [MLPClassifier((16,), max_iter=500, random_state=s).fit(P, ya).score(Q, yb) for s in range(5)]
        print(f"{nm:34s} {'unscaled' if raw else 'standardised'}: test accuracy over 5 seeds mean {np.mean(r):.3f}  min {min(r):.3f}  max {max(r):.3f}")
n0 = MLPClassifier((16,), max_iter=1, random_state=0).fit(Xa * big, ya); Z = Xa * big @ n0.coefs_[0]
import sys; sys.path.insert(0, "../../../ch31/code")
from mlp_scratch import MyMLP
mm = MyMLP(hidden=16, alpha=0.0, n_iter=0).fit(Xa * big, ya); G = mm.loss_grad(Xa * big, (ya[:, None] == np.unique(ya)) * 1.0)[1][0]
print("first-layer gradient at the start, row norms: x1 (thousands)", np.linalg.norm(G[0]).round(3), " x2", np.linalg.norm(G[1]).round(5), " ratio", round(np.linalg.norm(G[0]) / np.linalg.norm(G[1]), 0))
print("net input at the start: std of the x1 part", (Xa[:, [0]] * big[0] @ n0.coefs_[0][[0]]).std().round(1), " std of the x2 part", (Xa[:, [1]] @ n0.coefs_[0][[1]]).std().round(3))
print("fraction of hidden outputs that are 0 (ReLU off) at the start:", float((np.maximum(Z + n0.intercepts_[0], 0) == 0).mean()).__round__(3))

print("\n== Q4  adam: training loss falls, validation accuracy flat (the setting of Problem 31C.3)")
from sklearn.metrics import log_loss
Xd2, yd2 = make_moons(400, noise=0.5, random_state=3); Xd2 = StandardScaler().fit_transform(Xd2); Xt, yt, Xv, yv = Xd2[:60], yd2[:60], Xd2[60:], yd2[60:]
nn = MLPClassifier((100,), solver="adam", alpha=0, random_state=0, learning_rate_init=0.01, batch_size=60); vl, va, tl = [], [], []
for ep in range(300):
    nn.partial_fit(Xt, yt, classes=[0, 1]); tl.append(log_loss(yt, nn.predict_proba(Xt))); vl.append(log_loss(yv, nn.predict_proba(Xv))); va.append(nn.score(Xv, yv))
for ep in (10, 25, 50, 100, 200, 300): print(f"epoch {ep:3d}: training loss {tl[ep - 1]:.3f}  validation loss {vl[ep - 1]:.3f}  validation accuracy {va[ep - 1]:.3f}")
print("minimum validation loss at epoch", int(np.argmin(vl)) + 1, round(min(vl), 3), "; largest validation accuracy", round(max(va), 3), "at epoch", int(np.argmax(va)) + 1)

print("\n== Q5  diabetes: why a network does not beat least squares")
Xd, yd = load_diabetes(return_X_y=True); Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.25, random_state=0)
r2 = lambda yp: 1 - ((yte - yp) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
print("least squares test R2", round(r2(LinearRegression().fit(Xtr, ytr).predict(Xte)), 3), "  ridge(alpha=1)", round(r2(Ridge(1.0).fit(Xtr, ytr).predict(Xte)), 3))
sx = StandardScaler().fit(Xtr); A, B = sx.transform(Xtr), sx.transform(Xte); m, s = ytr.mean(), ytr.std()
print("weights of (16,): ", 10 * 16 + 16 + 16 + 1, " training rows:", len(ytr), "  weights per row:", round((10 * 16 + 16 + 16 + 1) / len(ytr), 2))
for a in (0.1, 10.0):
    t = [r2(MLPRegressor(hidden_layer_sizes=(16,), solver="lbfgs", alpha=a, max_iter=2000, random_state=sd).fit(A, (ytr - m) / s).predict(B) * s + m) for sd in range(10)]
    print(f"MLPRegressor(hidden_layer_sizes=(16,), lbfgs, alpha={a}): test R2 over 10 seeds mean {np.mean(t):.3f}  min {min(t):.3f}  max {max(t):.3f}")

print("\n== Q6  features learned without labels: a few labels, many unlabelled digits")
Xg, yg = load_digits(return_X_y=True); Xg = Xg / 16.0
Xtr, Xte, ytr, yte = train_test_split(Xg, yg, test_size=0.25, stratify=yg, random_state=0)
rbm = BernoulliRBM(100, learning_rate=0.1, n_iter=30, random_state=0).fit(Xtr)        # all 1347 training images, no labels
Htr, Hte = rbm.transform(Xtr), rbm.transform(Xte)
for nl in (50, 100, 300, 1347):
    a, b = [], []
    for sd in range(5):
        idx = np.random.default_rng(sd).permutation(len(ytr))[:nl]
        if len(np.unique(ytr[idx])) < 10: continue
        a.append(LogisticRegressionCV(Cs=6, cv=3, max_iter=2000, l1_ratios=(0,), scoring='accuracy', use_legacy_attributes=False).fit(Xtr[idx], ytr[idx]).score(Xte, yte)); b.append(LogisticRegressionCV(Cs=6, cv=3, max_iter=2000, l1_ratios=(0,), scoring='accuracy', use_legacy_attributes=False).fit(Htr[idx], ytr[idx]).score(Hte, yte))
    print(f"{nl:5d} labelled digits ({len(a)} draws): pixels {np.mean(a):.3f}   RBM features {np.mean(b):.3f}   difference {np.mean(b) - np.mean(a):+.3f}")
