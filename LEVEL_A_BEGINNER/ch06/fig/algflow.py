"""Algorithms 3.1-3.7 drawn as flowcharts whose boxes carry the numbered steps (Burden and Faires style)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from figstyle import *
from matplotlib.patches import FancyBboxPatch, Polygon
plt.rcParams.update({'mathtext.fontset': 'stix', 'font.family': 'STIXGeneral'})
MONO = 'DejaVu Sans Mono'; FS = 8.8; LH = 0.175; GAP = 0.27
XL = 0.8
LAB = 0.6

def P(step, lines, kind='proc', **kw):
    d = dict(step=step, lines=lines if isinstance(lines, list) else [lines], kind=kind); d.update(kw); return d

def tw(t, fs=FS, bold=False):
    f = plt.figure(); r = f.canvas.get_renderer()
    w = f.text(0, 0, t, fontsize=fs, weight='bold' if bold else 'normal').get_window_extent(renderer=r).width / f.dpi
    plt.close(f); return w

def draw(title, purpose, inputs, nodes, out):
    global XL, XR, SL, SR
    XL = 0.75 if any(n.get('back') for n in nodes) else 0.08
    mw = max([tw(t) for n in nodes for t in n['lines']] + [tw(t) for t in inputs] + [tw(t) - LAB + 0.05 for t in purpose] + [tw(title, FS + 0.4, True) - LAB])
    sw = max([tw(t) for n in nodes if n.get('side') for t in n['side']['lines']] + [tw(n['side'].get('fn', ''), 5.9) for n in nodes if n.get('side')] + [0.8])
    XR = XL + LAB + 0.08 + mw + 0.14; SL = XR + 0.25; SR = SL + sw + 0.2
    head = [('', title, 'title')] + [('', p, 'plain') for p in purpose] + [('INPUT' if k == 0 else '', t, '') for k, t in enumerate(inputs)]
    allnodes = [dict(step=None, rows=head, kind='start')] + nodes
    for n in allnodes:
        if 'rows' not in n: n['rows'] = [('Step %s' % n['step'] if k == 0 else '', t, '') for k, t in enumerate(n['lines'])]
        n['h'] = len(n['rows']) * LH + 0.12 + (0.12 if n.get('fn') else 0)
        if n.get('side'):
            sl = n['side']['lines']; n['sh'] = len(sl) * LH + 0.1 + (0.12 if n['side'].get('fn') else 0)
            n['h'] = max(n['h'], n['sh'])
    H = sum(n['h'] for n in allnodes) + GAP * (len(allnodes) - 1) + 0.1
    WF = SR + 0.55; fig = plt.figure(figsize=(WF, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, WF); ax.set_ylim(H, 0); ax.axis('off')
    y = 0.05
    for n in allnodes: n['top'] = y; n['bot'] = y + n['h']; n['mid'] = y + n['h'] / 2; y = n['bot'] + GAP
    idx = {n['step']: k for k, n in enumerate(allnodes) if n['step'] is not None}
    def line(pts, c=NAVY, head=True):
        xs, ys = zip(*pts); ax.plot(xs, ys, color=c, lw=0.8, zorder=1)
        if head: arr(ax, pts[-2], pts[-1], c=c, lw=0.8, ms=6)
    def lab(x, yy, t, ha='center'): ax.text(x, yy, t, fontsize=6.8, color=GREY, style='italic', ha=ha, va='center')
    def box(x0, x1, top, bot, kind):
        if kind == 'dec':
            e = 0.12
            ax.add_patch(Polygon([(x0, (top + bot) / 2), (x0 + e, top), (x1 - e, top), (x1, (top + bot) / 2), (x1 - e, bot), (x0 + e, bot)],
                                 fc='#FDF0DC', ec=OR, lw=0.9, zorder=2))
        else:
            fc, ec, rs = (LB, BLUE, 0.05)
            if kind in ('start', 'term'): fc, ec, rs = ('white', NAVY, 0.12)
            ls = (0, (3, 2)) if kind == 'note' else '-'
            if kind == 'note': fc, ec = ('#F5F6F8', GREY)
            ax.add_patch(FancyBboxPatch((x0, top), x1 - x0, bot - top, boxstyle='round,pad=0,rounding_size=%.2f' % rs,
                                        fc=fc, ec=ec, lw=0.9 if kind != 'start' else 1.1, ls=ls, zorder=2))
    def text_rows(x0, top, rows, labcol, fn=None, x1=None):
        for k, (l, t, st) in enumerate(rows):
            yy = top + 0.06 + LH * (k + 0.5)
            if st == 'title':
                ax.text(x0 + 0.08, yy, t, fontsize=FS + 0.4, weight='bold', color=NAVY, va='center', zorder=3); continue
            if st == 'plain':
                ax.text(x0 + 0.08, yy, t, fontsize=FS, color='black', va='center', zorder=3); continue
            if l: ax.text(x0 + 0.08, yy, l, fontsize=FS, weight='bold', color=NAVY, va='center', zorder=3)
            ax.text(x0 + (labcol if labcol else 0.08), yy, t, fontsize=FS, color='black', va='center', zorder=3)
        if fn: ax.text(x1 - 0.07, top + 0.06 + LH * len(rows) + 0.05, fn, fontsize=5.9, color=GREY, family=MONO, ha='right', va='center', zorder=3)
    lane_r = [SR + 0.15]; lane_l = [XL - 0.5]
    def jump(n, x_from, tgt_step, label):
        t = allnodes[idx[tgt_step]]; xr = lane_r[0]; lane_r[0] += 0.1
        gy = t['top'] - GAP / 2
        line([(x_from, n['mid']), (xr, n['mid']), (xr, gy), ((XL + XR) / 2 + 0.02, gy)], c=GREY)
        if label: lab((XR + SL) / 2, n['mid'] - 0.08, label)
    for k, n in enumerate(allnodes):
        kind = n['kind']; x0 = XL + (0.04 if kind == 'dec' else 0)
        box(XL, XR, n['top'], n['bot'], kind)
        lc = LAB + (0.06 if kind == 'dec' else 0) if kind != 'start' else 0.62
        text_rows(XL + (0.06 if kind == 'dec' else 0), n['top'] + (n['h'] - (len(n['rows']) * LH + 0.12 + (0.12 if n.get('fn') else 0))) / 2,
                  n['rows'], lc, n.get('fn'), XR)
        # fall-through arrow
        if k + 1 < len(allnodes) and not n.get('nofall') and allnodes[k + 1]['kind'] != 'note':
            XC = (XL + XR) / 2; line([(XC, n['bot']), (XC, allnodes[k + 1]['top'])])
            if kind == 'dec' or 'down' in n: lab((XL + XR) / 2 - 0.15 - (0.25 if kind != 'dec' else 0), n['bot'] + 0.1, n.get('down', 'no'))
        if n.get('back') is not None:
            t = allnodes[idx[n['back']]]; xl = lane_l[0]; lane_l[0] -= 0.09
            gy = t['top'] - GAP / 2
            line([(XL, n['mid']), (xl, n['mid']), (xl, gy), ((XL + XR) / 2 - 0.02, gy)], c=GREY)
            lab((xl + XL) / 2 + 0.04, n['mid'] - 0.09, n.get('backlab', ''))
        # side box
        sx1 = XR
        if n.get('side'):
            s = n['side']; st, sb = n['mid'] - n['sh'] / 2, n['mid'] + n['sh'] / 2
            box(SL, SR, st, sb, 'term' if s.get('stop') else 'proc')
            text_rows(SL, st - 0.01, [('', t, '') for t in s['lines']], 0, s.get('fn'), SR)
            line([(XR, n['mid']), (SL, n['mid'])]); lab((XR + SL) / 2, n['mid'] - 0.08, n.get('right', 'yes'))
            sx1 = SR
            if s.get('stop'): continue
            tgt = s.get('to')
            if tgt == n['step'] + 1 if isinstance(n['step'], int) else False:
                gy = n['bot'] + GAP / 2
                line([((SL + SR) / 2, sb), ((SL + SR) / 2, gy), ((XL + XR) / 2 + 0.02, gy)], c=GREY); continue
            jump(n, sx1, tgt, None)
        if n.get('to') is not None and not n.get('side'):
            jump(n, XR, n['to'], n.get('right', 'yes'))
    fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.03, facecolor='white'); plt.close(fig)

def _mk():
    pass

ALGS = {}
ALGS['alg31'] = ('ALGORITHM 3.1   Linear regression by the normal equations (fit)',
 [r'To find $\mathbf{w}$ and $b$ that minimise $J=\Sigma_i\,(y_i-\mathbf{x}_i^{T}\mathbf{w}-b)^2$ (Eq. 3.2-1).'],
 [r'$n\times p$ data $x_{ij}$; targets $y_1,\ldots,y_n$; fit_intercept.'],
 [P(1, [r'If fit_intercept, set $\bar x_j=\Sigma_i\,x_{ij}/n$ ($j=1,\ldots,p$) and $\bar y=\Sigma_i\,y_i/n$;',
        r'otherwise $\bar x_j=0$, $\bar y=0$.',
        r'Set $\tilde x_{ij}=x_{ij}-\bar x_j$ and $\tilde y_i=y_i-\bar y$.'], fn='x_mean, y_mean, Xc, yc'),
  P(2, [r'For $j,k=1,\ldots,p$ set $a_{jk}=\Sigma_i\, \tilde x_{ij}\,\tilde x_{ik}$.'], fn='A'),
  P(3, [r'For $j=1,\ldots,p$ set $c_j=\Sigma_i\, \tilde x_{ij}\,\tilde y_i$.'], fn='c'),
  P(4, [r'For $k=1,\ldots,p$ do Steps 5$-$7.'], fn='_solve'),
  P(5, [r'Find $r\geq k$ with the largest $|a_{rk}|$;', r'swap rows $r$ and $k$ of $A$ and of $\mathbf{c}$.']),
  P(6, [r'Is $|a_{kk}|\leq$ tol?'], 'dec', side=dict(lines=["OUTPUT ('dependent", "columns'); STOP."], stop=True)),
  P(7, [r'For $i=k+1,\ldots,p$ set $m=a_{ik}/a_{kk}$,', r'$a_{ij}=a_{ij}-m\,a_{kj}$ ($j=k,\ldots,p$) and $c_i=c_i-m\,c_k$.'],
    back=4, backlab=r'next $k$', down=r'after $k=p$'),
  P(8, [r'For $k=p,p-1,\ldots,1$ set $w_k=(c_k-\Sigma_{j>k}\,a_{kj}w_j)/a_{kk}$.'], fn='back substitution'),
  P(9, [r'Set $b=\bar y-\Sigma_j\, \bar x_j\,w_j$.'], fn='intercept_'),
  P(10, [r'OUTPUT (coef_ $=\mathbf{w}$, intercept_ $=b$); STOP.'], 'term')])
ALGS['alg32'] = ('ALGORITHM 3.2   Prediction and score',
 [r'To predict new targets with a fitted model, and to measure the fit by $R^2$.'],
 [r'$\mathbf{w}$, $b$; $m\times p$ data $z_{ij}$; for the score, true targets $y_1,\ldots,y_m$.'],
 [P(1, [r'For $i=1,\ldots,m$ set $\hat y_i=\Sigma_j\, z_{ij}\,w_j+b$.'], fn='predict'),
  P(2, [r'Set $\bar y=\Sigma_i\,y_i/m$, $SS_{\mathrm{res}}=\Sigma_i\,(y_i-\hat y_i)^2$,', r'$SS_{\mathrm{tot}}=\Sigma_i\,(y_i-\bar y)^2$.'], fn='score'),
  P(3, [r'Set $R^2=1-SS_{\mathrm{res}}/SS_{\mathrm{tot}}$.']),
  P(4, [r'OUTPUT ($\hat y_1,\ldots,\hat y_m$, $R^2$); STOP.'], 'term')])

if __name__ == '__main__':
    import types
    for name, (t, pu, inp, nodes) in ALGS.items():
        GAP = {'alg31': 0.42}.get(name, 0.27)
        draw(t, pu, inp, nodes, os.path.join(HERE, 'png', name + '.png'))
        print(name)
