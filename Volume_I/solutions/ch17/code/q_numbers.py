# Numbers quoted in the model answers to the questions of Chapter 17
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import SGDClassifier, RidgeClassifier, LogisticRegression, RidgeClassifierCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
np.set_printoptions(precision=4, suppress=True)
TX = np.array([[4, 3], [1, 0], [3, 4], [0, 1]], float); TY = np.array([1, -1, 1, -1.])
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
sc = StandardScaler().fit(Xtr); A = sc.transform(Xtr); s = 2 * ytr - 1.0

print("== Q1  which samples count")
w = np.array([2., 1.]); b = -3.0; z = TY * (TX @ w + b)
print("tiles, final perceptron line (2,1,-3): margins", z, " mistakes z<=0:", (z <= 0).sum(), " inside margin z<=1:", (z <= 1).sum())
w = np.zeros(2); b = 0.0; who = []
for e in range(3):
    for i in range(4):
        if TY[i] * (TX[i] @ w + b) <= 0: w = w + TY[i] * TX[i]; b += TY[i]; who.append(i + 1)
print("tiles that caused an update, in order:", who)
w = np.zeros(30); b = 0.0; touched = set(); ups = []
for e in range(200):
    m = 0
    for i in range(len(A)):
        if s[i] * (A[i] @ w + b) <= 0: w = w + s[i] * A[i]; b += s[i]; m += 1; touched.add(i)
    ups.append(m)
    if m == 0: break
print("cancer, perceptron 200 epochs: updates in epochs 1..5", ups[:5], " last", ups[-1], " distinct samples ever used", len(touched), "of", len(A))
h = SGDClassifier(alpha=1e-3, random_state=0).fit(A, ytr); zz = s * h.decision_function(A)
print("cancer, hinge SGD alpha=1e-3: n =", len(zz), " z<=0:", (zz <= 0).sum(), " z<=1:", (zz <= 1).sum(), " z>1:", (zz > 1).sum())

print("\n== Q2  decay at every step or once per epoch (Example 17.7-2: eta 0.1, alpha 1)")
def epoch(each):
    w = np.zeros(2); b = 0.0
    for i in range(4):
        z = TY[i] * (TX[i] @ w + b)
        if each: w = 0.9 * w
        if z <= 1: w = w + 0.1 * TY[i] * TX[i]; b += 0.1 * TY[i]
    return (w if each else 0.6 * w), b
print("every step:", *epoch(True), " once, factor 1 - 4*eta*alpha = 0.6 at the end:", *epoch(False))
print("per-step factor 0.9; product over 4 steps:", 0.9 ** 4, " vs 1 - 4*0.1 =", 1 - 0.4)
n = len(A); eta, al = 0.01, 1e-3
print("cancer scale: n =", n, " eta 0.01 alpha 1e-3: per step", 1 - eta * al, " per epoch (1-eta*alpha)^n =", round((1 - eta * al) ** n, 5), " wrongly applied once per epoch:", 1 - eta * al)

print("\n== Q3  one feature a thousand times larger")
for f in (1, 1000):
    A2 = A.copy(); A2[:, 0] *= f
    out = []
    for loss in ("hinge", "squared_hinge"):
        m = SGDClassifier(loss=loss, random_state=0).fit(A2, ytr); out.append(f"{loss}: max|w| = {np.abs(m.coef_).max():.3g}, n_iter_ = {m.n_iter_}")
    print("factor", f, " | ".join(out))
a_ = 1e-4; eta1 = a_ ** -0.25; x2 = 1000.0 ** 2
print("first optimal step", eta1, " margin gain eta*(|x|^2+1) with |x|^2 ~ 1e6:", f"{eta1 * (x2 + 1):.3g}", "; with scaled |x|^2 ~ 30:", eta1 * 31)
print("mean squared norm of a scaled sample:", round((A ** 2).sum(1).mean(), 2), " with the first column x1000:", f"{((A * np.r_[1000, np.ones(29)]) ** 2).sum(1).mean():.3g}")

print("\n== Q4  easy points and three losses")
x0 = np.array([1, 2, 3, 4, 5, 6.]); y0 = np.array([-1, -1, -1, 1, 1, 1]); far = np.array([12., 14., 16.])
for name, (x, yy) in {"six": (x0, y0), "nine": (np.r_[x0, far], np.r_[y0, np.ones(3)])}.items():
    X1 = x[:, None]; r = RidgeClassifier(alpha=1e-6).fit(X1, yy); lg = LogisticRegression(C=1e4, max_iter=10000).fit(X1, yy)
    sv = LinearSVC(C=1e4, max_iter=100000).fit(X1, yy)
    bd = lambda m: -m.intercept_[0] / m.coef_.ravel()[0]
    print(f"{name}: boundary ridge {bd(r):.3f}  logistic {bd(lg):.3f}  hinge (LinearSVC) {bd(sv):.3f}   ridge accuracy {r.score(X1, yy):.3f}")
zz = 3.0
print("force at z = 3: ridge 2(z-1) =", 2 * (zz - 1), " log loss |L'| =", round(1 / (1 + np.exp(zz)), 4), " hinge 0")

print("\n== Q5  a stream that changes character")
def stream(avg, restart=False, seed=0, pre=30, post=150, nb=20):
    rng = np.random.RandomState(seed)
    def batch(sign):
        yb = rng.randint(0, 2, nb); return rng.randn(nb, 2) + np.where(yb[:, None] == 1, sign, -sign) * np.array([1.5, 0.]), yb
    mk = lambda: SGDClassifier(loss="log_loss", alpha=1e-3, average=avg, random_state=0)
    m = mk(); acc = []; low = 0
    for k in range(pre + post):
        sign = 1 if k < pre else -1
        Xb, yb = batch(sign)
        acc.append(np.mean(m.predict(Xb) == yb) if k > 0 else 0.5)    # test before training on the batch
        if restart and k >= pre and acc[-1] < 0.5:
            low += 1
            if low == 3: m = mk(); low = 0                              # three bad batches: start again
        m.partial_fit(Xb, yb, classes=[0, 1])
    return np.array(acc)
def back(acc, pre=30):   # first batch after the change at which a 3-batch mean reaches 0.8
    for k in range(pre, len(acc) - 2):
        if acc[k:k + 3].mean() >= 0.8: return k - pre + 1
    return None
for name, kw in (("plain", dict(avg=False)), ("average=True", dict(avg=True)), ("average=True, restart after 3 bad batches", dict(avg=True, restart=True))):
    r = [back(stream(seed=sd, **kw)) for sd in range(5)]
    print(f"{name:42s} batches to recover (5 seeds): {r}")
print("updates before the change: 30 batches x 20 =", 600)

print("\n== Q6  ridge or logistic on the breast-cancer training half (5-fold, as Section 17.13)")
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for name, est in (("RidgeClassifierCV", RidgeClassifierCV(alphas=np.logspace(-2, 3, 12))), ("LogisticRegression", LogisticRegression())):
    sc_ = cross_val_score(make_pipeline(StandardScaler(), est), Xtr, ytr, cv=cv); print(f"{name:20s} {sc_.mean():.4f} +- {sc_.std():.4f}")
