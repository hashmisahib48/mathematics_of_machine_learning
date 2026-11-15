# Equation and matrix images of Chapter 14. Run: python3 fig/eqs.py
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq, matimg
import matplotlib.pyplot as plt
T=r'^{\mathrm{T}}'; X=r'\mathbf{X}'; y=r'\mathbf{y}'; w=r'\mathbf{w}'
E={
'e1431': r'$\ln\mu_i \;=\; \eta_i \;=\; b + \mathbf{w}^{\mathrm{T}}\mathbf{x}_i\,, \qquad \mu_i \;=\; e^{\,\eta_i} \;=\; \exp\,(b + \mathbf{w}^{\mathrm{T}}\mathbf{x}_i)$',
'e1432': r'$\mathrm{Var}\,(y_i) \;=\; \phi\,\mu_i^{\,p}$',
'e1433': [r'$P(y\,|\,\mu) \;=\; \dfrac{e^{-\mu}\,\mu^{\,y}}{y!}\,, \qquad \ell(\mu) \;=\; y\ln\mu-\mu-\ln y!$'],
'e1434': r'$\dfrac{\partial \ell}{\partial \mu} \;=\; \dfrac{y-\mu}{\phi\,\mu^{\,p}}$',
'e1435': r'$d(y,\mu) \;=\; 2\int_{\mu}^{y}\dfrac{y-t}{t^{\,p}}\,dt$',
'e1436': r'$d(y,\mu) \;=\; 2\left[\,y\ln\dfrac{y}{\mu}-(y-\mu)\,\right]$',
'e1437': r'$d(y,\mu) \;=\; 2\left[\,\dfrac{y-\mu}{\mu}-\ln\dfrac{y}{\mu}\,\right]$',
'e1438': r'$d(y,\mu) \;=\; 2\left[\,\dfrac{y^{\,2-p}}{(1-p)(2-p)}-\dfrac{y\,\mu^{\,1-p}}{1-p}+\dfrac{\mu^{\,2-p}}{2-p}\,\right]$',
'e1439': r'$J(\mathbf{w},b) \;=\; \dfrac{1}{2\sum_i s_i}\sum_{i=1}^{n} s_i\,d(y_i,\mu_i) \;+\; \dfrac{\alpha}{2}\,\Vert\mathbf{w}\Vert^{2}$',
'e14311': [r'$h_i \;=\; \tilde s_i\,\mu_i^{\,1-p}\,\left[\,\mu_i+(p-1)(y_i-\mu_i)\,\right]$',
           r'$\mathbf{H} \;=\; \tilde{\mathbf{X}}^{\mathrm{T}}\,\mathrm{diag}(h_1,\ldots,h_n)\,\tilde{\mathbf{X}} \;+\; \alpha\,\mathrm{diag}(1,\ldots,1,0)$'],
'e14312': r'$\mathbf{g} \;=\; \mathbf{X}^{\mathrm{T}}(\mathbf{y}-\boldsymbol{\mu})\,, \qquad \mathbf{H} \;=\; \mathbf{X}^{\mathrm{T}}\mathbf{M}\,\mathbf{X}\,, \qquad \mathbf{M}=\mathrm{diag}(\mu_1,\ldots,\mu_n)$',
'e14313': r'$\boldsymbol{\beta}_{\mathrm{new}} \;=\; \boldsymbol{\beta}+\boldsymbol{\delta}\,, \qquad \mathbf{H}\,\boldsymbol{\delta} \;=\; \mathbf{g}\,, \qquad \boldsymbol{\beta}=(b,\,\mathbf{w})$',
'e14314': r'$\boldsymbol{\beta}_{\mathrm{new}} \;=\; (\mathbf{X}^{\mathrm{T}}\mathbf{M}\,\mathbf{X})^{-1}\mathbf{X}^{\mathrm{T}}\mathbf{M}\,\mathbf{z}\,, \qquad z_i \;=\; \eta_i+\dfrac{y_i-\mu_i}{\mu_i}$',
'e14315': r'$D^{2} \;=\; 1-\dfrac{\sum_i s_i\,d(y_i,\hat\mu_i)}{\sum_i s_i\,d(y_i,\bar y)}$',
'e14316': r'$d(N,\,e\lambda) \;=\; e\;d\!\left(\dfrac{N}{e},\,\lambda\right)$',
'e1491': r'$f(y;\theta,\phi) \;=\; \exp\!\left(\dfrac{y\,\theta-\kappa(\theta)}{\phi}+c(y,\phi)\right)$',
'e1492': r'$\mathrm{E}\,[y] \;=\; \kappa^{\prime}(\theta) \;=\; \mu\,, \qquad \mathrm{Var}\,(y) \;=\; \phi\,\kappa^{\prime\prime}(\theta) \;=\; \phi\,V(\mu)$',
'e1493': r'$\sum_{i=1}^{n} s_i\,(y_i-\mu_i)\,\mathbf{x}_i \;=\; \mathbf{0}$',
}
# gradient (two lines, second is long)
E['e14310']=[r'$\dfrac{\partial J}{\partial b} \;=\; -\sum_{i}\tilde s_i\,(y_i-\mu_i)\,\mu_i^{\,1-p}$',
             r'$\dfrac{\partial J}{\partial w_j} \;=\; -\sum_{i}\tilde s_i\,(y_i-\mu_i)\,\mu_i^{\,1-p}\,x_{ij} \;+\; \alpha\,w_j$']
mkeq({k:v for k,v in E.items() if k!='e14310' and k!='e14311'},'.')
mkeq({'e14310':E['e14310'],'e14311':E['e14311']},'.')
M=lambda rows:[[str(v) for v in r] for r in rows]
ex=json.load(open('verify/example_numbers.json'))
s1=ex['step1']; s2=ex['step2']
matimg('m1441',[r'$\mathbf{g}=$',M([[0],[12]]),r'$,\quad \mathbf{H}=3\,\mathbf{X}^{\mathrm{T}}\mathbf{X}=$',M([[15,30],[30,90]])])
matimg('m1442',[r'$\boldsymbol{\delta}=\mathbf{H}^{-1}\mathbf{g}=\frac{1}{450}$',M([[90,-30],[-30,15]]),M([[0],[12]]),r'$=\frac{1}{450}$',M([[-360],[180]]),'$=$',M([['-0.8'],['0.4']])])
f=lambda v:f'{v:.4f}'
matimg('m1443',[r'$\mathbf{g}=$',M([[f(v)] for v in s2['g']]),r'$,\quad \mathbf{H}=$',M([[f(v) for v in r] for r in s2['H']]),r'$,\quad \boldsymbol{\delta}=$',M([[f(v)] for v in s2['delta']])])
print('eqs ok')
