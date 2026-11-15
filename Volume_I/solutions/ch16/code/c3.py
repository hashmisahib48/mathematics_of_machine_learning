# Problem 16C.3: hours or minutes with the default C = 1.
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from figstyle import *
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sig = lambda z: 1 / (1 + np.exp(-z))
hours = np.arange(1., 9); y = np.array([0, 0, 0, 1, 0, 1, 1, 1.])
free = LogisticRegression(C=np.inf, tol=1e-10, max_iter=1000).fit(hours[:, None], y); wf, bf = float(free.coef_[0, 0]), float(free.intercept_[0])
print("unpenalised: b = %.4f  w = %.4f per hour  p(6 h) = %.4f" % (bf, wf, sig(bf + 6 * wf)))
res = {}
for nm, f in (("hours", 1.0), ("minutes", 60.0)):
    m = LogisticRegression(C=1.0, tol=1e-10, max_iter=10000).fit(f * hours[:, None], y); w, b = float(m.coef_[0, 0]), float(m.intercept_[0]); res[nm] = (m, f)
    print(f"{nm:8s}: w = {w:.5f} per {nm[:-1]}  = {w * f:.4f} per hour   b = {b:.4f}   p(6 h) = {m.predict_proba([[6 * f]])[0, 1]:.4f}"
          f"   penalty w^2/2 = {w * w / 2:.6f}   log loss of the training set = {np.sum(np.logaddexp(0, b + w * f * hours) - y * (b + w * f * hours)):.4f}")
print("ratio of the squared weights (the penalty) for the same slope per hour: 60^2 =", 60 ** 2)
for nm, f in (("hours", 1.0), ("minutes", 60.0)):
    sc = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, tol=1e-10, max_iter=10000)).fit(f * hours[:, None], y)
    print(f"{nm:8s} after StandardScaler: p(6 h) = {sc.predict_proba([[6 * f]])[0, 1]:.4f}   weight of the scaled feature = {sc[-1].coef_[0, 0]:.4f}   scale_ = {sc[0].scale_[0]:.3f}")
Cs = np.logspace(-3, 7, 41); eff = {k: [] for k in ("hours", "minutes", "scaled")}
for C in Cs:
    for nm, f in (("hours", 1.0), ("minutes", 60.0)):
        m = LogisticRegression(C=C, tol=1e-10, max_iter=10000).fit(f * hours[:, None], y); eff[nm].append(m.coef_[0, 0] * f)
    sc = make_pipeline(StandardScaler(), LogisticRegression(C=C, tol=1e-10, max_iter=10000)).fit(hours[:, None], y); eff["scaled"].append(sc[-1].coef_[0, 0] / hours.std())
fig, ax = plt.subplots(figsize=(3.4, 2.0), dpi=240)
ax.semilogx(Cs, eff["hours"], color=GREEN, lw=1.7, label='hours'); ax.semilogx(Cs, eff["minutes"], color=OR, lw=1.5, ls='--', label='minutes')
ax.semilogx(Cs, eff['scaled'], color=PUR, lw=1.4, ls='-.', label='standardised'); ax.axhline(wf, color=GREY, lw=0.8, ls=':'); ax.text(2e-3, wf + 0.04, 'unpenalised 1.282', fontsize=8, color=GREY)
ax.set_xlabel('$C$'); ax.set_ylabel('weight per hour'); ax.legend(fontsize=8, loc='lower right', handlelength=1.8); ax.set_ylim(0, 1.45)
fig.tight_layout(); fs = min(t_.get_fontsize() for t_ in fig.findobj(matplotlib.text.Text) if t_.get_text().strip()); save(fig, 'fig/s16c3.png'); print("smallest text", fs)
