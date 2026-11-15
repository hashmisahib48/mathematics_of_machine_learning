# Problem 15C.3: sine and cosine against a periodic spline for a time of day.
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from figstyle import *
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer
def circ(h, c): d = np.abs(h - c); return np.minimum(d, 24 - d)
demand = lambda h: 2 + 5*np.exp(-circ(h, 8)**2/4) + 3*np.exp(-circ(h, 17.5)**2/6)     # cyclic.py
smooth = lambda h: 4 + 2.5*np.cos(2*np.pi*(h - 15)/24)                                # one hump a day
hh = np.linspace(0, 24, 241)[:-1]                                                     # 0, 0.1, ..., 23.9
def fourier(h, K): return np.column_stack([g(2*np.pi*k*h/24) for k in range(1, K+1) for g in (np.sin, np.cos)])
def r2(f, Z, h): m = LinearRegression().fit(Z(h), f(h)); return 1 - ((f(h) - m.predict(Z(h)))**2).sum() / ((f(h) - f(h).mean())**2).sum(), m
sp = SplineTransformer(knots=np.linspace(0, 24, 9)[:, None], extrapolation='periodic').fit(hh[:, None])
print("columns: sin/cos 2, two harmonics 4, three harmonics 6, periodic spline", sp.transform(hh[:, None]).shape[1])
for nm, f in (("two rush hours (cyclic.py)", demand), ("one hump a day", smooth)):
    print(nm)
    for lab, Z in (("sin, cos", lambda h: fourier(h, 1)), ("2 harmonics", lambda h: fourier(h, 2)), ("3 harmonics", lambda h: fourier(h, 3)),
                   ("periodic spline", lambda h: sp.transform(h[:, None]))):
        R, m = r2(f, Z, hh); e = np.abs(f(hh) - m.predict(Z(hh))).max()
        print(f"   {lab:16s} R2 {R:.4f}   largest error {e:.3f}")
m1 = LinearRegression().fit(fourier(hh, 1), demand(hh)); a, b = m1.coef_
print("the one-harmonic fit is  c + A sin + B cos  with  A = %.3f, B = %.3f: one peak at %.2f h and one trough at %.2f h, amplitude %.3f"
      % (a, b, (np.degrees(np.arctan2(a, b)) % 360) / 15, ((np.degrees(np.arctan2(a, b)) % 360) / 15 + 12) % 24, np.hypot(a, b)))
print("true demand: peaks at 8 h (%.2f) and 17.5 h (%.2f), valley at 13 h (%.2f), night 3 h (%.2f)" % tuple(demand(np.array([8., 17.5, 13., 3.]))))
print("one-harmonic fit at 8, 13, 17.5, 3 h:", np.round(m1.predict(fourier(np.array([8., 13., 17.5, 3.]), 1)), 2))
fig, axs = plt.subplots(1, 2, figsize=(5.4, 1.9), dpi=240)
for ax, f, ttl in ((axs[0], demand, 'two rush hours'), (axs[1], smooth, 'one hump a day')):
    ax.plot(hh, f(hh), color=GREY, lw=1.0, ls=':', label='true')
    for K, col, ls in ((1, OR, '--'), (2, PUR, '-.')):
        m = LinearRegression().fit(fourier(hh, K), f(hh)); ax.plot(hh, m.predict(fourier(hh, K)), color=col, lw=1.3, ls=ls, label=f'{K} harmonic' + ('s' if K > 1 else ''))
    m = LinearRegression().fit(sp.transform(hh[:, None]), f(hh)); ax.plot(hh, m.predict(sp.transform(hh[:, None])), color=GREEN, lw=1.4, label='periodic spline')
    ax.set_xticks([0, 6, 12, 18, 24]); ax.set_xlabel('hour ' + '(' + ttl + ')'); ax.set_ylabel('demand'); ax.set_ylim(0, 11)
axs[0].legend(fontsize=7.6, loc='upper center', ncol=2, columnspacing=0.8, handlelength=1.6)
fig.tight_layout(); fs = min(t_.get_fontsize() for t_ in fig.findobj(matplotlib.text.Text) if t_.get_text().strip()); save(fig, 'fig/s15c3.png'); print("smallest text", fs)
