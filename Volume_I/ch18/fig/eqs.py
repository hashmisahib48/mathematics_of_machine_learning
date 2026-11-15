# Equation and matrix images of Chapter 18. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq, matimg
import matplotlib.pyplot as plt
T=r'^{\mathrm{T}}'; Si=r'\Sigma^{-1}'; mu=r'\mathbf{\mu}'; x=r'\mathbf{x}'; w=r'\mathbf{w}'; a=r'\mathbf{a}'; d=r'\mathbf{d}'
E={
'e1831': rf'$P(k\,|\,{x}) \;=\; \dfrac{{\pi_k\,p({x}\,|\,k)}}{{\sum_{{j=1}}^{{K}}\pi_j\,p({x}\,|\,j)}}$',
'e1832': rf'$p({x}\,|\,k) \;=\; \dfrac{{1}}{{(2\pi)^{{p/2}}\sqrt{{\det\Sigma_k}}}}\,\exp\!\left(-\dfrac{{1}}{{2}}\,({x}-{mu}_k){T}\,\Sigma_k^{{-1}}\,({x}-{mu}_k)\right)$',
'e1833': rf'$\delta_k({x}) \;=\; \ln\pi_k \;-\; \dfrac{{1}}{{2}}\ln\det\Sigma_k \;-\; \dfrac{{1}}{{2}}\,({x}-{mu}_k){T}\,\Sigma_k^{{-1}}\,({x}-{mu}_k)$',
'e1834x': [rf'$({x}-{mu}_k){T}{Si}({x}-{mu}_k) \;=\; {x}{T}{Si}{x} \;-\; 2\,{mu}_k{T}{Si}{x} \;+\; {mu}_k{T}{Si}{mu}_k$',
           rf'$\delta_k({x}) \;=\; c({x}) \;+\; {mu}_k{T}{Si}{x} \;-\; \dfrac{{1}}{{2}}\,{mu}_k{T}{Si}{mu}_k \;+\; \ln\pi_k\,$',
           rf'$c({x}) \;=\; -\dfrac{{1}}{{2}}\ln\det\Sigma \;-\; \dfrac{{1}}{{2}}\,{x}{T}{Si}{x}$'],
'e1835': rf'$\delta_k({x}) \;=\; {w}_k{T}{x} + b_k\,, \qquad {w}_k = {Si}{mu}_k\,, \qquad b_k = -\dfrac{{1}}{{2}}\,{mu}_k{T}{Si}{mu}_k + \ln\pi_k$',
'e1836': rf'$P(k\,|\,{x}) \;=\; \dfrac{{e^{{\delta_k({x})}}}}{{\sum_{{j=1}}^{{K}} e^{{\delta_j({x})}}}}$',
'e1837x': [rf'$\ln\dfrac{{P(1\,|\,{x})}}{{P(0\,|\,{x})}} \;=\; {w}{T}{x} + b\,, \qquad {w} = {Si}({mu}_1-{mu}_0)$',
           rf'$b \;=\; -\dfrac{{1}}{{2}}\,({mu}_0+{mu}_1){T}{Si}({mu}_1-{mu}_0) \;+\; \ln\dfrac{{\pi_1}}{{\pi_0}}$'],
'e1838': rf'$\hat k \;=\; \arg\max_k\left[\,\ln\pi_k \;-\; \dfrac{{1}}{{2}}\,d_M^{{\,2}}({x},{mu}_k)\,\right]\,, \qquad d_M^{{\,2}}({x},{mu}_k) = ({x}-{mu}_k){T}{Si}({x}-{mu}_k)$',
'e1839': rf'$x^{{*}} \;=\; \dfrac{{\mu_0+\mu_1}}{{2}} \;+\; \dfrac{{\sigma^2}}{{\mu_1-\mu_0}}\,\ln\dfrac{{\pi_0}}{{\pi_1}}$',
'e18310': rf'$\pi_k=\dfrac{{n_k}}{{n}}\,, \qquad {mu}_k=\dfrac{{1}}{{n_k}}\sum_{{i\in k}}{x}_i\,, \qquad \Sigma \;=\; \dfrac{{1}}{{n}}\sum_{{k=1}}^{{K}}\sum_{{i\in k}}({x}_i-{mu}_k)({x}_i-{mu}_k){T}$',
'e1841': rf'$J({a}) \;=\; \dfrac{{\left({a}{T}({mu}_1-{mu}_0)\right)^2}}{{{a}{T}\Sigma\,{a}}}$',
'e1842x': [rf'$\nabla J \;=\; \dfrac{{2\,{a}{T}{d}}}{{{a}{T}\Sigma{a}}}\left[\,{d} \;-\; \dfrac{{{a}{T}{d}}}{{{a}{T}\Sigma{a}}}\,\Sigma{a}\right] \;=\; \mathbf{{0}} \quad\Longrightarrow\quad \Sigma\,{a} \;\propto\; {d}\,, \qquad {d}={mu}_1-{mu}_0$',
           rf'${a}_{{\mathrm{{F}}}} \;\propto\; {Si}\,({mu}_1-{mu}_0)$'],
'e1843x': [rf'$J({a}) \;=\; \dfrac{{{a}{T}S_b\,{a}}}{{{a}{T}S_w\,{a}}}\,, \qquad S_w=\Sigma\,,$',
           rf'$S_b \;=\; \sum_{{k=1}}^{{K}}\pi_k\,({mu}_k-\bar{{{mu}}})({mu}_k-\bar{{{mu}}}){T}\,, \qquad \bar{{{mu}}}=\sum_k\pi_k{mu}_k$'],
'e1844': rf'$S_b\,{a} \;=\; \lambda\,S_w\,{a}\,, \qquad {a}{T}S_w\,{a}=1$',
'e1845': rf'$\mathbf{{z}} \;=\; \mathbf{{A}}{T}({x}-\bar{{{mu}}})\,, \qquad \mathrm{{ratio}}_j \;=\; \dfrac{{\lambda_j}}{{\lambda_1+\cdots+\lambda_m}}$',
'e1851': rf'$-\dfrac{{1}}{{2}}\,{x}{T}\!\left(\Sigma_k^{{-1}}-\Sigma_l^{{-1}}\right){x} \;+\; \left(\Sigma_k^{{-1}}{mu}_k-\Sigma_l^{{-1}}{mu}_l\right){T}{x} \;+\; c_{{kl}} \;=\; 0$',
'e1852': rf'$\hat\Sigma_k \;=\; \dfrac{{1}}{{n_k}}\sum_{{i\in k}}({x}_i-{mu}_k)({x}_i-{mu}_k){T}$',
'e1853': rf'$\Sigma_a \;=\; (1-a)\,\hat\Sigma \;+\; a\,\dfrac{{\mathrm{{tr}}\,\hat\Sigma}}{{p}}\,\mathbf{{I}}\,, \qquad 0\leq a\leq 1$',
'e1854': rf'$\Sigma_k(r) \;=\; (1-r)\,\hat\Sigma_k \;+\; r\,\mathbf{{I}}\,, \qquad 0\leq r\leq 1$',
'e1855': rf'$\Sigma_{{\mathrm{{auto}}}} \;=\; (1-\hat a)\,\hat\Sigma \;+\; \hat a\,\mathrm{{diag}}(\hat\Sigma)$',
'e1861x': [r'$a_2 x^2 + a_1 x + a_0 \;=\; 0\,, \qquad a_2=\dfrac{1}{2v_0}-\dfrac{1}{2v_1}\,, \quad a_1=\dfrac{\mu_1}{v_1}-\dfrac{\mu_0}{v_0}\,,$',
           r'$a_0=\dfrac{\mu_0^2}{2v_0}-\dfrac{\mu_1^2}{2v_1}-\dfrac{1}{2}\ln\dfrac{v_1}{v_0}+\ln\dfrac{\pi_1}{\pi_0}$'],
}
Ls={k:E.pop(k) for k in list(E) if k.endswith('x')}
mkeq(E,'.')
for k,v in Ls.items():
    n=len(v); fig=plt.figure(figsize=(.01,.01)); fig.text(0,0,'\n'.join(v),fontsize=11.5,linespacing=2.7)
    fig.savefig(k[:-1]+'.png',dpi=300,bbox_inches='tight',pad_inches=0.02,facecolor='white'); plt.close(fig)
M=lambda rows:[[str(v) for v in r] for r in rows]
matimg('m1862',[r'$\Sigma=\frac{1}{8}$',M([[6,4],[4,6]]),'$=$',M([['0.75','0.50'],['0.50','0.75']]),r'$,\quad \det\Sigma=\frac{20}{64}$'])
matimg('m1863',[r'$\Sigma^{-1}=\frac{8}{20}$',M([[6,-4],[-4,6]]),'$=$',M([['2.4','-1.6'],['-1.6','2.4']])])
matimg('m1864',[r'$\mathbf{w}=\Sigma^{-1}(\mathbf{\mu}_1-\mathbf{\mu}_0)=$',M([['2.4','-1.6'],['-1.6','2.4']]),M([[3],[1]]),'$=$',M([['5.6'],['-2.4']])])
print('eqs ok')
