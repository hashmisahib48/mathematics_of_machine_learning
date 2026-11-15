# Problem 31C.2: ReLU and tanh networks beyond the data. Writes fig/s31c2.png, prints numbers.
import sys, os, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
warnings.simplefilter('ignore', ConvergenceWarning)

rng = np.random.default_rng(1); xt = np.sort(rng.uniform(-3, 3, 40)); yt = np.sin(xt) + rng.normal(0, 0.1, 40)
fits = {}
for act in ('relu', 'tanh'):
    best = None
    for s in range(10):
        m = MLPRegressor(hidden_layer_sizes=(10,), activation=act, solver='lbfgs', alpha=1e-3, max_iter=3000, random_state=s).fit(xt[:, None], yt)
        if best is None or m.loss_ < best.loss_: best = m
    fits[act] = best
pts = [-20, -10, -6, -3, 0, 3, 6, 10, 20, 100]
print("x          :", pts)
for act, m in fits.items():
    print(f"{act:5s} f(x)  :", [round(float(m.predict([[v]])[0]), 2) for v in pts], f"  training rmse {np.sqrt(2 * m.loss_):.3f}")
for act, m in fits.items():
    f = lambda v: float(m.predict([[v]])[0])
    print(f"{act:5s} slope between 10 and 20: {(f(20) - f(10)) / 10:.4f}   between 20 and 100: {(f(100) - f(20)) / 80:.4f}   between -20 and -10: {(f(-10) - f(-20)) / 10:.4f}")
m = fits['relu']
w, b, v = m.coefs_[0][0], m.intercepts_[0], m.coefs_[1][:, 0]
kinks = np.sort(-b / w); print("ReLU kinks at x =", kinks.round(2).tolist(), " (units: %d)" % len(w))
m2 = fits['tanh']; w2, b2, v2 = m2.coefs_[0][0], m2.intercepts_[0], m2.coefs_[1][:, 0]
print("tanh limit at +inf = b_out + sum v sign(w) =", round(float(m2.intercepts_[1][0] + (v2 * np.sign(w2)).sum()), 3), "; at -inf =", round(float(m2.intercepts_[1][0] - (v2 * np.sign(w2)).sum()), 3))
xg = np.linspace(-12, 12, 400)
fig, ax = plt.subplots(figsize=(4.0, 2.0), dpi=240); ax.axvspan(-3, 3, color=LB, zorder=0, lw=0); ax.scatter(xt, yt, s=7, color=BLUE, zorder=4)
ax.plot(xg, fits['relu'].predict(xg[:, None]), color=GREEN, lw=1.5, label='ReLU'); ax.plot(xg, fits['tanh'].predict(xg[:, None]), color=PUR, lw=1.5, ls='--', label='tanh')
ax.plot(xg, np.sin(xg), color=GREY, lw=0.8, ls=':', label='sin $x$')
ax.set_ylim(-5, 5); ax.set_xlabel('$x$'); ax.set_ylabel('prediction'); ax.legend(fontsize=8.5, loc='upper left', handlelength=1.6, labelspacing=0.2, borderaxespad=0.1); ax.tick_params(labelsize=8.5); ax.grid(color=GRID, lw=0.5)
print("min font size:", min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()))
save(fig, 'fig/s31c2.png')
