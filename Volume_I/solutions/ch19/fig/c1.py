# Figure S19.1 and numbers for Problem 19C.1: r copies of feature 1 in GaussianNB, query (3, 6)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from figstyle import *
from sklearn.naive_bayes import GaussianNB
X = np.array([[1, 2], [2, 4], [3, 3], [4, 5], [6, 9], [8, 7]], float); y = np.array([0, 0, 0, 1, 1, 1]); q = np.array([[3., 6.]])
R = np.arange(1, 9); P1 = []; LO = []
for r in R:
    Xr = np.c_[X, np.tile(X[:, [0]], (1, r - 1))]; qr = np.c_[q, np.tile(q[:, [0]], (1, r - 1))]
    g = GaussianNB().fit(Xr, y); j = g.predict_joint_log_proba(qr)[0]; LO.append(j[1] - j[0]); P1.append(g.predict_proba(qr)[0, 1])
    print(f"copies of x1 r = {r}: log-odds {LO[-1]:8.4f}  P(class 1) = {P1[-1]:.4f}  label {g.predict(qr)[0]}")
d = LO[0] - LO[1]; print("each extra copy changes the log-odds by", round(-d, 4), "; first r with label 0:", int(R[np.array(LO) < 0][0]))
fig, ax = plt.subplots(figsize=(3.3, 1.8), dpi=240)
ax.plot(R, P1, 'o-', color=BLUE, lw=1.5, ms=4); ax.axhline(0.5, color=GREY, ls='--', lw=0.9)
ax.set_xlabel('copies of feature 1', fontsize=9.5); ax.set_ylabel('P(class 1 | x)', fontsize=9.5); ax.set_ylim(-0.02, 1.02); ax.tick_params(labelsize=9)
ax.text(8, 0.53, 'decision threshold', fontsize=9, color=GREY, ha='right', va='bottom'); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
save(fig, 'fig/s19c1.png')
