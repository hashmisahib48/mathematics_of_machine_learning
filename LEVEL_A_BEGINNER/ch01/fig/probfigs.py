import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from figstyle import *
def base(w=2.6, h=1.55):
    fig, ax = plt.subplots(figsize=(w, h), dpi=300); ax.tick_params(labelsize=7, length=2.5); return fig, ax
fig, ax = base()
ax.plot([10, 20, 30, 40, 50], [2.1, 2.9, 4.2, 4.8, 6.1], 'o', ms=4, color=BLUE)
ax.set_xlabel('temperature $x$ (°C)', fontsize=7.5); ax.set_ylabel('output $y$ (mV)', fontsize=7.5)
ax.set_xlim(0, 60); ax.set_ylim(0, 7)
for xx, yy in zip([10, 20, 30, 40, 50], [2.1, 2.9, 4.2, 4.8, 6.1]): ax.text(xx, yy + 0.35, f'{yy}', fontsize=6.5, ha='center', color=NAVY)
fig.tight_layout(pad=0.2); fig.savefig(os.path.join(HERE, 'png', 'p31.png'), dpi=300, facecolor='white'); plt.close(fig)
fig, ax = base()
xs = np.linspace(1, 9, 14); rng = np.random.default_rng(4); r = 0.9 * (xs - 5) ** 2 / 8 - 0.9 + rng.normal(0, 0.12, xs.size)
ax.axhline(0, color=GREY, lw=0.7); ax.plot(xs, r, 'o', ms=3.6, color=RED)
ax.set_xlabel('feature $x$', fontsize=7.5); ax.set_ylabel('error $e_i$', fontsize=7.5)
fig.tight_layout(pad=0.2); fig.savefig(os.path.join(HERE, 'png', 'p36.png'), dpi=300, facecolor='white'); plt.close(fig)
