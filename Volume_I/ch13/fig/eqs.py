# Equation images of Chapter 13. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq
import matplotlib.pyplot as plt
T=r'^{\mathrm{T}}'; w=r'\mathbf{w}'; xi=r'\mathbf{x}_i'
E={
'e1331': [r'$H_\epsilon(z) \;=\; z^2 \qquad\qquad\qquad\;\; (|z| \leq \epsilon)$', r'$H_\epsilon(z) \;=\; 2\epsilon\,|z| - \epsilon^2 \qquad (|z| > \epsilon)$'],
'e1332': rf'$J({w},b,\sigma) \;=\; \sum_{{i=1}}^{{n}}\left[\,\sigma + \sigma\,H_\epsilon\!\left(\dfrac{{r_i}}{{\sigma}}\right)\right] \;+\; \alpha\,\Vert {w}\Vert^2\,, \qquad r_i = y_i - b - {w}{T}{xi}$',
'e1333': [r'$\sigma + \sigma H_\epsilon(r_i/\sigma) \;=\; \sigma + \dfrac{r_i^{\,2}}{\sigma} \qquad\qquad\qquad (|r_i| \leq \epsilon\sigma)$', r'$\sigma + \sigma H_\epsilon(r_i/\sigma) \;=\; \sigma + 2\epsilon\,|r_i| - \epsilon^2\sigma \qquad (|r_i| > \epsilon\sigma)$'],
'e1334': r'$\omega_i \;=\; \min\!\left(1,\; \dfrac{\epsilon\,\sigma}{|r_i|}\right)$',
'e1335': rf'$\nabla_{{{w}}} J \;=\; -\dfrac{{2}}{{\sigma}}\sum_{{i=1}}^{{n}} \omega_i\, r_i\, {xi} \;+\; 2\alpha\,{w}\,, \qquad \dfrac{{\partial J}}{{\partial b}} \;=\; -\dfrac{{2}}{{\sigma}}\sum_{{i=1}}^{{n}} \omega_i\, r_i$',
'e1336': rf'$\left(\mathbf{{A}}{T}\boldsymbol{{\Omega}}\mathbf{{A}} + \alpha\,\mathbf{{P}}\right)\boldsymbol{{\theta}} \;=\; \mathbf{{A}}{T}\boldsymbol{{\Omega}}\,\mathbf{{y}}\,, \qquad \boldsymbol{{\theta}} = \binom{{b}}{{{w}}}$',
'e1337': r'$\dfrac{\partial J}{\partial \sigma} \;=\; n - k\,\epsilon^2 - \dfrac{1}{\sigma^2}\sum_{|r_i|\leq\epsilon\sigma} r_i^{\,2} \;=\; 0$',
'e1338': r'$\sigma^2 \;=\; \dfrac{\sum_{|r_i|\leq\epsilon\sigma} r_i^{\,2}}{n - k\,\epsilon^2}$',
'e1341': r'$t \;=\; \mathrm{median}_i\,\left|\,y_i - \mathrm{median}(y)\,\right|$',
'e1342': r'$\left(1 - f^{\,s}\right)^{N} \;\leq\; 1-\gamma \qquad\Longrightarrow\qquad N \;=\; \left\lceil \dfrac{\ln(1-\gamma)}{\ln\left(1 - f^{\,s}\right)} \right\rceil$',
'e1343': r'$w_{ij} \;=\; \dfrac{y_j - y_i}{x_j - x_i}\,, \qquad b_{ij} \;=\; y_i - w_{ij}\,x_i$',
'e1344': r'$\hat w \;=\; \mathrm{median}_{\,i<j}\; w_{ij}\,, \qquad \hat b \;=\; \mathrm{median}_{\,i}\,\left(y_i - \hat w\,x_i\right)$',
'e1345': r'$\hat{\boldsymbol{\theta}} \;=\; \arg\min_{\boldsymbol{\theta}} \sum_{S}\, \Vert \boldsymbol{\theta}_S - \boldsymbol{\theta} \Vert\,, \qquad \boldsymbol{\theta}_S = (b_S,\, w_S)$',
'e1346': r'$\left(1 - \varepsilon^{*}\right)^{s} \;=\; \dfrac{1}{2} \qquad\Longrightarrow\qquad \varepsilon^{*} \;=\; 1 - \left(\dfrac{1}{2}\right)^{1/s}$',
'e1351': [r'$\rho_\tau(r) \;=\; \tau\, r \qquad\qquad\;\;\; (r \geq 0)$', r'$\rho_\tau(r) \;=\; (\tau - 1)\, r \qquad\; (r < 0)$'],
'e1352': rf'$J({w},b) \;=\; \dfrac{{1}}{{n}}\sum_{{i=1}}^{{n}} \rho_\tau\!\left(y_i - b - {w}{T}{xi}\right) \;+\; \alpha \sum_{{j=1}}^{{p}} |w_j|$',
'e1353': r'$\dfrac{dJ}{dc} \;=\; -\tau\,\dfrac{N_{\mathrm{above}}}{n} \;+\; (1-\tau)\,\dfrac{N_{\mathrm{below}}}{n} \;=\; 0 \qquad\Longrightarrow\qquad \dfrac{N_{\mathrm{below}}}{n} \;=\; \tau$',
'e1354': [r'$\min\;\; \dfrac{1}{n}\sum_{i=1}^{n}\left(\tau\,u_i + (1-\tau)\,v_i\right) + \alpha\sum_{j=1}^{p}\left(s_j + t_j\right)$', r'$\mathrm{subject\ to}\;\; y_i - b - \mathbf{x}_i^{\mathrm{T}}(\mathbf{s}-\mathbf{t}) \;=\; u_i - v_i\,, \qquad u_i,\, v_i,\, s_j,\, t_j \;\geq\; 0$'],
'e13111': r'$p(\varepsilon) \;=\; \dfrac{1}{2s}\exp\!\left(-\dfrac{|\varepsilon|}{s}\right)\,, \qquad \ln L \;=\; -n\ln(2s) \;-\; \dfrac{1}{s}\sum_{i=1}^{n}\,|y_i - b - \mathbf{w}^{\mathrm{T}}\mathbf{x}_i|$',
}
mkeq(E,'.')
from mkeq import matimg
M=lambda rows:[[str(v) for v in r] for r in rows]
matimg('m1361',[M([['3.7962','9.2680'],['9.2680','27.4889']]),M([['b'],['w']]),'$=$',M([['19.6568'],['57.2942']])])
print('eqs ok')
