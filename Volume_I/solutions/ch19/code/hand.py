# Hand checks for Problems 19A.1-19A.4 (every intermediate number of the written solutions)
import numpy as np
np.set_printoptions(precision=4, suppress=True)
words = ["free", "lunch", "meeting", "now", "prize", "report", "win"]
Nh = np.array([0, 2, 3, 1, 0, 2, 0.]); Ns = np.array([3, 0, 0, 3, 2, 0, 2.])
print("== 19A.1  'lunch now now', alpha = 1")
for nm, Nk in (("ham", Nh), ("spam", Ns)):
    den = Nk.sum() + 7; th = (Nk + 1) / den
    print(f"{nm}: N_k = {Nk.sum():.0f}, denominator {den:.0f}, theta = {np.round(th * den).astype(int)}/{den:.0f}")
th_h = (Nh + 1) / 15; th_s = (Ns + 1) / 17
jh = 0.5 * th_h[1] * th_h[3] ** 2; js = 0.5 * th_s[1] * th_s[3] ** 2
print(f"ham:  0.5 * {th_h[1]:.4f} * {th_h[3]:.4f}^2 = {jh:.6f}   (= 0.5*3*2^2/15^3 = {0.5*3*4/15**3:.6f})")
print(f"spam: 0.5 * {th_s[1]:.4f} * {th_s[3]:.4f}^2 = {js:.6f}   (= 0.5*1*4^2/17^3 = {0.5*16/17**3:.6f})")
print(f"P(spam) = {js/(jh+js):.4f}  P(ham) = {jh/(jh+js):.4f}")
print("log ratio per word (spam - ham): lunch", round(np.log(th_s[1] / th_h[1]), 4), " now", round(np.log(th_s[3] / th_h[3]), 4), " sum with now twice", round(np.log(th_s[1] / th_h[1]) + 2 * np.log(th_s[3] / th_h[3]), 4))

print("\n== 19A.2  Gaussian, query (3.5, 4)")
X = np.array([[1, 2], [2, 4], [3, 3], [4, 5], [6, 9], [8, 7]], float); y = np.array([0, 0, 0, 1, 1, 1]); q = np.array([3.5, 4.0])
sc = []
for k in (0, 1):
    A = X[y == k]; mu = A.mean(0); v = A.var(0); print(f"class {k}: mean {mu}  variance (1/n) {v}")
    t = -0.5 * (np.log(2 * np.pi * v) + (q - mu) ** 2 / v); print("   ln p(x_j|k) =", t, " sum", t.sum().round(4), " + ln 0.5 =", np.log(0.5).round(4)); sc.append(np.log(0.5) + t.sum())
    print("   squared distance to mean:", ((q - mu) ** 2).sum())
    print("   parts: ln(2 pi v) =", np.log(2 * np.pi * v), " (x-mu)^2/v =", (q - mu) ** 2 / v)
print("scores", np.round(sc, 4), " difference", round(sc[0] - sc[1], 4), " P(class 0) =", round(1 / (1 + np.exp(sc[1] - sc[0])), 4), " e^diff =", round(np.exp(sc[0] - sc[1]), 4))

print("\n== 19A.3  Bernoulli")
B = np.array([[1,0,0,1,1,0,0],[1,0,0,0,1,0,1],[1,0,0,1,0,0,1],[0,0,1,0,0,1,0],[0,1,1,1,0,0,0],[0,1,1,0,0,1,0]]); yb = y[::-1] if False else np.array([1,1,1,0,0,0])
Cs = B[yb == 1].sum(0); Ch = B[yb == 0].sum(0); print("document counts spam", Cs, "ham", Ch)
ts = (Cs + 1) / 5; th = (Ch + 1) / 5; print("theta spam", ts, "\ntheta ham ", th)
xb = np.array([1, 0, 1, 1, 0, 0, 0])
def sk(t): return np.log(0.5) + (xb * np.log(t) + (1 - xb) * np.log(1 - t)).sum()
ls, lh = sk(ts), sk(th); print("scores spam, ham", round(ls, 4), round(lh, 4), " joint", np.exp(ls).round(6), np.exp(lh).round(6), " P(spam)", round(1 / (1 + np.exp(lh - ls)), 4))
print("contributions of absent words, spam:", np.round(np.log(1 - ts) * (1 - xb), 4), "sum", round((np.log(1 - ts) * (1 - xb)).sum(), 4), " ham:", np.round(np.log(1 - th) * (1 - xb), 4), "sum", round((np.log(1 - th) * (1 - xb)).sum(), 4))
print("present words, spam:", np.round(np.log(ts) * xb, 4), round((np.log(ts) * xb).sum(), 4), " ham:", np.round(np.log(th) * xb, 4), round((np.log(th) * xb).sum(), 4))
# multinomial for comparison, 'free meeting now'
xm = np.array([1, 0, 1, 1, 0, 0, 0]); tms = (Ns + 1) / 17; tmh = (Nh + 1) / 15
a = np.log(.5) + xm @ np.log(tms); b = np.log(.5) + xm @ np.log(tmh); print("multinomial scores", round(a, 4), round(b, 4), "P(spam)", round(1 / (1 + np.exp(b - a)), 4))

print("\n== 19A.4  categorical smoothing")
cnt = np.array([1, 2, 1.])
for m in (3, 4):
    c = np.r_[cnt, np.zeros(m - 3)]; print(f"m = {m}: denominator {4 + m}, theta =", (c + 1) / (4 + m), " sum", ((c + 1) / (4 + m)).sum())
print("without smoothing category 3 would be 0 and a test sample with category 3 would get score -inf")

print("\n== 19C.1  duplicated column (hand): the score of the copied feature")
rng = np.random.default_rng(0); Xs = np.array([[1,2],[2,4],[3,3],[4,5],[6,9],[8,7]], float)
from sklearn.naive_bayes import GaussianNB
g1 = GaussianNB().fit(Xs, y); X2 = np.c_[Xs, Xs[:, 0]]; g2 = GaussianNB().fit(X2, y)
for qq in ([3, 6], [3.5, 4], [2, 3]):
    qq = np.array([qq], float); q2 = np.c_[qq, qq[:, 0]]
    j1 = g1.predict_joint_log_proba(qq)[0]; j2 = g2.predict_joint_log_proba(q2)[0]
    t1 = -0.5 * (np.log(2*np.pi*g1.var_[:,0]) + (qq[0,0]-g1.theta_[:,0])**2/g1.var_[:,0])
    print(f"query {qq[0]}: joint log (orig) {j1.round(4)}  (dup) {j2.round(4)}  added term {(j2-j1).round(4)} = ln p(x_1|k) {t1.round(4)}; P(class1) {g1.predict_proba(qq)[0,1]:.4f} -> {g2.predict_proba(q2)[0,1]:.4f}; labels {g1.predict(qq)[0]} {g2.predict(q2)[0]}")
print("log-odds change: orig", round(g1.predict_joint_log_proba(np.array([[3,6.]]))[0,1]-g1.predict_joint_log_proba(np.array([[3,6.]]))[0,0],4), " dup", round(g2.predict_joint_log_proba(np.array([[3,6,3.]]))[0,1]-g2.predict_joint_log_proba(np.array([[3,6,3.]]))[0,0],4))

print("\n== 19C.2  parameters")
for p in (1, 2, 5, 10, 20, 30):
    print(f"p = {p:2d}: full table {2*(2**p-1)+1:>14,d}  naive {2*p+1:>4d}   cells per class {2**p:>13,d}")
print("30 binary features, a sample of one million: cells", 2**30, " samples per cell", round(1e6 / 2**30, 5))

print("\n== 19A.2 extra: a point where the two rules disagree")
g = GaussianNB().fit(X, y)
for pt in ([0.0, 9.0], [4.0, 4.5], [5.0, 5.0]):
    pt = np.array([pt]); d = ((pt - g.theta_) ** 2).sum(1)
    print("point", pt[0], "squared distances to the means", d, " nearest mean:", d.argmin(), " GaussianNB:", g.predict(pt)[0], " P(class 1) =", g.predict_proba(pt)[0, 1].round(4))

print("\n== 19A.1 slip: 'now' counted once")
a = 0.5 * (1 / 17) * (4 / 17); b = 0.5 * (3 / 15) * (2 / 15); print(f"spam {a:.4f} ham {b:.4f} P(spam) {a/(a+b):.4f}")
