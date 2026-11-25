import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from figstyle import *
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
P = lambda n: os.path.join(HERE, 'png', n)
def fin(fig, n): fig.tight_layout(pad=0.3); fig.savefig(P(n), dpi=300, facecolor='white'); plt.close(fig)
def sty(ax, xl, yl): ax.set_xlabel(xl, fontsize=7); ax.set_ylabel(yl, fontsize=7); ax.tick_params(labelsize=6.3, length=2.2)
# 8.2-1 iris petal length, prior x density
I = load_iris(); x = I.data[:, 2]; y = I.target; g = GaussianNB().fit(x[:, None], y); t = np.linspace(0.5, 7.5, 400)
fig, ax = plt.subplots(figsize=(2.05, 1.6), dpi=300)
for k, col, nm in zip(range(3), [BLUE, OR, RED], ['setosa', 'versicolor', 'virginica']):
    m, v = g.theta_[k, 0], g.var_[k, 0]; ax.plot(t, g.class_prior_[k] * np.exp(-(t - m) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v), color=col, lw=1.1, label=nm)
    ax.plot(x[y == k], np.full((y == k).sum(), -0.04 - 0.03 * k), '|', ms=3, color=col)
pr = g.predict(t[:, None]); 
for i in np.where(np.diff(pr) != 0)[0]: ax.axvline(t[i], color=GREY, lw=0.5, ls='--')
ax.legend(fontsize=5.6, frameon=False); sty(ax, 'petal length (cm)', r'$P(c)\,p(x\mid c)$'); fin(fig, 'f81.png')
# 8.4-1 example
A = np.array([[2, 6], [3, 8], [4, 7.]]); Bp = np.array([[6, 3], [7, 5], [8, 4], [7, 4.]])
g = GaussianNB(var_smoothing=0).fit(np.r_[A, Bp], [0, 0, 0, 1, 1, 1, 1])
a, b = np.meshgrid(np.linspace(0.5, 9.5, 300), np.linspace(1.5, 9.5, 300)); Z = g.predict_proba(np.c_[a.ravel(), b.ravel()])[:, 0].reshape(a.shape)
fig, ax = plt.subplots(figsize=(2.05, 1.85), dpi=300)
ax.contour(a, b, Z, levels=[0.5], colors=[GREY], linewidths=0.8, linestyles='--')
for k, col in [(0, BLUE), (1, OR)]:
    m, v = g.theta_[k], g.var_[k]; pdf = np.exp(-(a - m[0]) ** 2 / (2 * v[0]) - (b - m[1]) ** 2 / (2 * v[1]))
    ax.contour(a, b, pdf, levels=[0.05, 0.3, 0.7], colors=[col], linewidths=0.5)
ax.plot(A[:, 0], A[:, 1], 'o', ms=3.5, color=BLUE); ax.plot(Bp[:, 0], Bp[:, 1], 's', ms=3.5, color=OR); ax.plot(5, 5, '*', ms=8, color=RED)
ax.text(5.15, 5.15, 'z', fontsize=6.5, color=RED); ax.text(2.0, 8.6, 'A', fontsize=7, color=BLUE); ax.text(8.3, 4.8, 'B', fontsize=7, color=OR)
ax.set_aspect('equal'); sty(ax, '$x_1$', '$x_2$'); fin(fig, 'f82.png')
# 8.7-1 confidence histograms
X, Y = load_breast_cancer(return_X_y=True); a_, b_, c_, d_ = train_test_split(X, Y, test_size=0.25, random_state=0, stratify=Y)
pn = GaussianNB().fit(a_, c_).predict_proba(b_)[:, 1]; pl = make_pipeline(StandardScaler(), LogisticRegression()).fit(a_, c_).predict_proba(b_)[:, 1]
fig, axs = plt.subplots(2, 1, figsize=(2.05, 2.4), dpi=300, sharex=True); bins = np.linspace(0, 1, 21)
for ax, p, nm in zip(axs, [pn, pl], ['naive Bayes', 'logistic regression']):
    ax.hist(p[d_ == 0], bins, color=RED, alpha=0.75, label='malignant'); ax.hist(p[d_ == 1], bins, color=BLUE, alpha=0.6, label='benign')
    ax.set_title(nm, fontsize=7, color=NAVY); sty(ax, '', 'tumors'); ax.set_yscale('log')
axs[0].legend(fontsize=5.6, frameon=False, loc='upper center'); sty(axs[1], 'predicted P(benign)', 'tumors'); fin(fig, 'f83.png')
# P8-1 digit templates
D = load_digits(); m = MultinomialNB().fit(D.data, D.target)
fig, axs = plt.subplots(2, 5, figsize=(2.05, 0.95), dpi=300)
for k, ax in enumerate(axs.flat): ax.imshow(np.exp(m.feature_log_prob_[k]).reshape(8, 8), cmap='gray_r'); ax.axis('off'); ax.set_title(str(k), fontsize=6, pad=1)
fin(fig, 'pj81.png'); print('ok')
