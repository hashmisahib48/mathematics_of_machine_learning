# Equation and matrix images of Chapter 16. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq, matimg
import matplotlib.pyplot as plt
T=r'^{\mathrm{T}}'; X=r'\mathbf{X}'; y=r'\mathbf{y}'; w=r'\mathbf{w}'; p=r'\mathbf{p}'; xv=r'\mathbf{x}'
E={
# 16.2
'e1621': r'$\mathrm{odds} \;=\; \dfrac{p}{1-p}$',
'e1622': r'$z \;=\; \ln\dfrac{p}{1-p} \;=\; \ln(\mathrm{odds})$',
'e1623': r'$p \;=\; \sigma(z) \;=\; \dfrac{1}{1+e^{-z}}\,, \qquad \dfrac{p}{1-p} \;=\; e^{z}$',
# 16.3
'e1631': rf'$P(y=1\,|\,{xv}) \;=\; p \;=\; \sigma(b+{w}{T}{xv}) \;=\; \dfrac{{1}}{{1+e^{{-(b+{w}{T}{xv})}}}}$',
'e1632': [rf'$\ln\dfrac{{p}}{{1-p}} \;=\; b + w_1x_1 + w_2x_2 + \cdots + w_px_p$', r'$\dfrac{\mathrm{odds\ after}\ x_j\ \mathrm{grows\ by\ 1}}{\mathrm{odds\ before}} \;=\; e^{w_j}$'],
'e1633': r'$P(y_i\,|\,\mathbf{x}_i) \;=\; p_i^{\,y_i}\,(1-p_i)^{1-y_i}\,, \qquad y_i\in\{0,1\}$',
'e1634': r'$\ln L \;=\; \sum_{i=1}^{n}\left[\,y_i\ln p_i + (1-y_i)\ln(1-p_i)\right]$',
'e1635': [r'$J(\mathbf{w},b) \;=\; -\ln L \;=\; \sum_{i=1}^{n}\ell_i\,, \qquad \ell_i \;=\; -y_i\ln p_i-(1-y_i)\ln(1-p_i)$', r'$\ell_i \;=\; \ln\left(1+e^{-s_i z_i}\right)\,, \qquad s_i=2y_i-1\in\{-1,+1\}\,,\quad z_i=b+\mathbf{w}^{\mathrm{T}}\mathbf{x}_i$'],
'e1636': [r'$\dfrac{\partial \ell}{\partial p} \;=\; -\dfrac{y}{p}+\dfrac{1-y}{1-p}\,, \qquad \dfrac{\partial p}{\partial z} \;=\; p(1-p)$', r'$\dfrac{\partial \ell}{\partial z} \;=\; \left(-\dfrac{y}{p}+\dfrac{1-y}{1-p}\right)p(1-p) \;=\; -y(1-p)+(1-y)\,p \;=\; p-y$'],
'e1637': rf'$\nabla J \;=\; {X}{T}({p}-{y})\,, \qquad \dfrac{{\partial J}}{{\partial w_j}} \;=\; \sum_{{i=1}}^{{n}}(p_i-y_i)\,x_{{ij}}$',
'e1638': rf'$\mathbf{{H}} \;=\; \sum_{{i=1}}^{{n}} p_i(1-p_i)\,\mathbf{{x}}_i\mathbf{{x}}_i{T} \;=\; {X}{T}\mathbf{{S}}{X}\,, \qquad \mathbf{{S}}=\mathrm{{diag}}\left(p_1(1-p_1),\ldots,p_n(1-p_n)\right)$',
'e1639': r'$\mathbf{u}^{\mathrm{T}}\mathbf{H}\,\mathbf{u} \;=\; \sum_{i=1}^{n} p_i(1-p_i)\,\left(\mathbf{x}_i^{\mathrm{T}}\mathbf{u}\right)^2 \;\geq\; 0$',
'e16310': rf'${w}_{{\mathrm{{new}}}} \;=\; {w}-\mathbf{{H}}^{{-1}}\nabla J \;=\; {w}-({X}{T}\mathbf{{S}}{X})^{{-1}}{X}{T}({p}-{y})$',
'e16311': rf'${w}_{{\mathrm{{new}}}} \;=\; ({X}{T}\mathbf{{S}}{X})^{{-1}}{X}{T}\mathbf{{S}}\,\mathbf{{z}}\,, \qquad \mathbf{{z}} \;=\; {X}{w}+\mathbf{{S}}^{{-1}}({y}-{p})$',
'e16312': rf'$b+{w}{T}{xv} \;=\; 0\,, \qquad p \;=\; \sigma\left(\Vert{w}\Vert\,d\right)\,, \qquad d \;=\; \dfrac{{b+{w}{T}{xv}}}{{\Vert{w}\Vert}}$',
# 16.4
'e1641': r'$f_k \;=\; b_k+\mathbf{w}_k^{\mathrm{T}}\mathbf{x}\,, \qquad P(y=k\,|\,\mathbf{x}) \;=\; \dfrac{e^{f_k}}{\sum_{j=1}^{K}e^{f_j}}$',
'e1642': r'$J \;=\; -\sum_{i=1}^{n}\ln P(y_i\,|\,\mathbf{x}_i) \;=\; \sum_{i=1}^{n}\left[\ln\sum_{j=1}^{K}e^{f_{ij}} - f_{i,y_i}\right]$',
'e1643': r'$\dfrac{\partial J}{\partial \mathbf{w}_k} \;=\; \sum_{i=1}^{n}(P_{ik}-Y_{ik})\,\mathbf{x}_i\,, \qquad \dfrac{\partial J}{\partial b_k} \;=\; \sum_{i=1}^{n}(P_{ik}-Y_{ik})$',
'e1644': r'$P(y=1) \;=\; \dfrac{e^{f_1}}{e^{f_0}+e^{f_1}} \;=\; \dfrac{1}{1+e^{-(f_1-f_0)}} \;=\; \sigma(f_1-f_0)$',
# 16.5
'e1651': r'$J(\mathbf{w},b) \;=\; C\sum_{i=1}^{n}\ell_i \;+\; \rho\,\Vert\mathbf{w}\Vert_1 \;+\; \dfrac{1-\rho}{2}\,\Vert\mathbf{w}\Vert_2^{2}$',
'e1652': r'$\dfrac{J}{Cn} \;=\; \dfrac{1}{n}\,\sum_{i=1}^{n}\ell_i \;+\; \lambda\left[\rho\,\Vert\mathbf{w}\Vert_1+\dfrac{1-\rho}{2}\Vert\mathbf{w}\Vert_2^{2}\right]\,, \qquad \lambda \;=\; \dfrac{1}{Cn}$',
'e1653': r'$\sum_{i=1}^{n}\ell_i + \dfrac{1}{2C}\Vert\mathbf{w}\Vert_2^{2} \;=\; \dfrac{J}{C}\,, \qquad \mathbf{w}\sim\mathcal{N}(\mathbf{0},\,C\,\mathbf{I})$',
'e1654': [r'$C\,\mathbf{X}^{\mathrm{T}}(\mathbf{p}-\mathbf{y}) + (1-\rho)\,\mathbf{w} + \rho\,\mathbf{v} \;=\; \mathbf{0}\,, \qquad \sum_{i=1}^{n}(p_i-y_i) \;=\; 0$', r'$v_j=\mathrm{sign}(w_j)\ \ \mathrm{if}\ w_j\neq 0\,, \qquad v_j\in[-1,1]\ \ \mathrm{if}\ w_j=0$'],
}
E0=dict(E)
for k in ('e1632','e1635','e1636','e1654'): E0.pop(k)
mkeq(E0,'.')
def multi(name,lines,ls=2.6):
    fig=plt.figure(figsize=(.01,.01)); fig.text(0,0,'\n'.join(lines),fontsize=11.5,linespacing=ls); fig.savefig(name+'.png',dpi=300,bbox_inches='tight',pad_inches=0.02,facecolor='white'); plt.close(fig)
for k in ('e1632','e1635','e1636','e1654'): multi(k,E[k],3.3 if k=='e1636' else 2.6)
M=lambda rows:[[str(v) for v in r] for r in rows]
Xm=[[1,k] for k in range(1,9)]; ym=[[v] for v in [0,0,0,1,0,1,1,1]]
matimg('m1661',[rf'${X}=$',M(Xm),rf'$,\quad {y}=$',M(ym),rf'$,\quad {w}=$',M([['b'],['w']])])
matimg('m1662',[rf'$\mathbf{{H}}=\frac{{1}}{{4}}{X}{T}{X}=\frac{{1}}{{4}}$',M([[8,36],[36,204]]),'$=$',M([[2,9],[9,51]])])
matimg('m1663',[r'$\mathbf{H}^{-1}=\frac{1}{2\times 51-9\times 9}$',M([[51,-9],[-9,2]]),r'$=\frac{1}{21}$',M([[51,-9],[-9,2]])])
matimg('m1664',[r'$\mathbf{d}=-\mathbf{H}^{-1}\mathbf{g}=\frac{1}{21}$',M([[51,-9],[-9,2]]),M([[0],[7]]),r'$=\frac{1}{21}$',M([[-63],[14]]),'$=$',M([['-3'],['0.6667']])])
print('eqs ok')
