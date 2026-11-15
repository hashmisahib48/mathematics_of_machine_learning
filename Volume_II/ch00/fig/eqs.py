# Equation images of the refresher. Run: python3 fig/eqs.py
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq, matimg
import matplotlib.pyplot as plt
T = r'^{\mathrm{T}}'; X = r'\mathbf{X}'; y = r'\mathbf{y}'; w = r'\mathbf{w}'
E = {
 'e0031': rf'$\nabla J({w}) \;=\; -2\,{X}{T}({y}-{X}{w})\,, \qquad \mathbf{{H}} \;=\; 2\,{X}{T}{X}$',
 'e0042': [r'$\mu_{b|a} \;=\; \mu_b + \Sigma_{ba}\,\Sigma_{aa}^{-1}\,(\mathbf{x}_a-\mu_a)$',
           r'$\Sigma_{b|a} \;=\; \Sigma_{bb} - \Sigma_{ba}\,\Sigma_{aa}^{-1}\,\Sigma_{ab}$'],
 'e0051': r'$L(\mathbf{w},\boldsymbol{\lambda}) \;=\; f(\mathbf{w}) + \sum_{i=1}^{m}\lambda_i\,g_i(\mathbf{w})$',
 'e0053': r'$q(\boldsymbol{\lambda}) \;=\; \min_{\mathbf{w}}\,L(\mathbf{w},\boldsymbol{\lambda}) \;\leq\; f(\mathbf{v}) \qquad \mathrm{for\ every\ feasible\ }\mathbf{v}\ \mathrm{and\ every\ }\boldsymbol{\lambda}\geq\mathbf{0}$',
 'e0081': [r'$\ell \;=\; -\ln P_{y} \;=\; \ln\sum_{j=1}^{K} e^{f_j} \;-\; f_{y}$',
           r'$\dfrac{\partial\ell}{\partial f_k} \;=\; P_k - Y_k\,, \qquad Y_k=1\ \mathrm{if}\ k=y,\ \ \mathrm{else}\ 0$'],
}
mkeq(E, '.')
# the four KKT conditions, with labels in a second column
rows = [(r'$\nabla f(\mathbf{w}) + \sum_{i=1}^{m}\lambda_i\,\nabla g_i(\mathbf{w}) \;=\; \mathbf{0}$', 'stationarity'),
        (r'$g_i(\mathbf{w}) \;\leq\; 0 \quad \mathrm{for\ every}\ i$', 'primal feasibility'),
        (r'$\lambda_i \;\geq\; 0 \quad \mathrm{for\ every}\ i$', 'dual feasibility'),
        (r'$\lambda_i\,g_i(\mathbf{w}) \;=\; 0 \quad \mathrm{for\ every}\ i$', 'complementary slackness')]
fig = plt.figure(figsize=(4.1, 1.5)); n = len(rows)
for j, (f_, lab) in enumerate(rows):
    yy = 1 - (j + 0.5) / n
    fig.text(0.0, yy, f_, fontsize=11.5, va='center'); fig.text(0.585, yy, lab, fontsize=10.5, va='center', style='italic', color='#1F5FBF')
fig.savefig('e0052.png', dpi=300, bbox_inches='tight', pad_inches=0.02, facecolor='white'); plt.close(fig)
# block covariance matrix of the partitioned normal
matimg('m0041', [r'$\mathbf{x}\;=\;$', [[r'\mathbf{x}_a'], [r'\mathbf{x}_b']], r'$\;\sim\;\mathcal{N}\,\left(\right.$', [[r'\mu_a'], [r'\mu_b']],
                 r'$,\;$', [[r'\Sigma_{aa}', r'\Sigma_{ab}'], [r'\Sigma_{ba}', r'\Sigma_{bb}']], r'$\left.\right)$'], rh=0.24)
print('eqs ok')
