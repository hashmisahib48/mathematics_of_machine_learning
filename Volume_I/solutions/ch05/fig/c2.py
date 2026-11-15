# Problem 5C.2: the solution and the multiplier as the radius t grows. Writes fig/s5c2.png, verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
from scipy.optimize import minimize
c = np.array([3., 4.]); f = lambda w: np.sum((w - c) ** 2)
print("  t   |w|   w (constraint |w| <= t)    lambda   (equality formula 5/t-1)   min f")
rows = []
for t in (0.5, 1.0, 2.5, 4.0, 5.0, 6.0, 8.0):
    res = minimize(f, [0.1, 0.1], constraints=[{"type": "ineq", "fun": lambda w, t=t: t**2 - w @ w}], method="SLSQP", options={"ftol": 1e-12})
    w = res.x; lam = max(0.0, 5 / t - 1)                       # lambda = 5/t - 1 (Eq. 5.7-5); 0 once the ball holds c
    eq_lam = 5 / t - 1                                         # the same formula for the equality constraint |w| = t
    print(f"{t:4.1f} {np.linalg.norm(w):5.3f} {str(w.round(3)):16s} {lam:6.3f} (equality: {eq_lam:+.3f})   {res.fun:7.3f}")
    rows.append((t, lam, res.fun))
fig, axs = plt.subplots(1, 3, figsize=(5.2, 1.75), dpi=240)
th = np.linspace(0, 2 * np.pi, 300); ax = axs[0]
for r in (1, 2.5, 5, 6.5): ax.plot(r * np.cos(th), r * np.sin(th), color=PUR if r != 5 else RED, lw=0.8 if r != 5 else 1.0)
for r in (1, 2.5): ax.plot(*(c * r / 5), 'o', color=GREEN, ms=3)
ax.plot(*c, '*', color=NAVY, ms=7); ax.set_aspect('equal'); ax.set_xlim(-7, 7); ax.set_ylim(-7, 7); ax.set_xlabel('$w_1$', fontsize=8); ax.set_ylabel('$w_2$', fontsize=8)
ax.tick_params(labelsize=7.5); ax.set_title('circles $|w|=t$', fontsize=8)
tt = np.linspace(0.3, 8, 300); ax = axs[1]
ax.plot(tt, np.maximum(0, 5 / tt - 1), color=BLUE, lw=1.4); ax.plot(tt, 5 / tt - 1, color=GREY, lw=0.8, ls='--'); ax.set_ylim(-0.6, 5); ax.axvline(5, color=RED, lw=0.7, ls=':')
ax.set_xlabel('radius $t$', fontsize=8); ax.set_ylabel('multiplier $\\lambda$', fontsize=8); ax.tick_params(labelsize=7.5); ax.grid(color=GRID, lw=0.5)
ax = axs[2]; ax.plot(tt, (5 - np.minimum(tt, 5)) ** 2, color=GREEN, lw=1.4); ax.axvline(5, color=RED, lw=0.7, ls=':')
ax.set_xlabel('radius $t$', fontsize=8); ax.set_ylabel('minimum of $f$', fontsize=8); ax.tick_params(labelsize=7.5); ax.grid(color=GRID, lw=0.5)
fig.tight_layout(w_pad=0.8); save(fig, 'fig/s5c2.png')
