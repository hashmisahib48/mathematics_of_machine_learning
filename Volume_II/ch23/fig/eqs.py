# Equation and matrix images of Chapter 23. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq, matimg
import matplotlib.pyplot as plt
T=r'^{\mathrm{T}}'; w=r'\mathbf{w}'; x=r'\mathbf{x}'
E={
'e2321': r'$L_{\varepsilon}(r) \;=\; \max\left(0,\;|r|-\varepsilon\right)\,, \qquad r \;=\; y-f(\mathbf{x})$',
'e2331': rf'$J(\mathbf{{w}},b) \;=\; \dfrac{{1}}{{2}}\Vert\mathbf{{w}}\Vert^2 + C\sum_{{i=1}}^{{n}} L_{{\varepsilon}}\left(y_i-\mathbf{{w}}{T}\mathbf{{x}}_i-b\right)$',
'e2336': r'$\max_{\boldsymbol{\beta}}\;\; -\dfrac{1}{2}\boldsymbol{\beta}^{\mathrm{T}}\mathbf{K}\boldsymbol{\beta} + \mathbf{y}^{\mathrm{T}}\boldsymbol{\beta} - \varepsilon\sum_{i=1}^{n}|\beta_i|\,, \qquad \sum_{i=1}^{n}\beta_i = 0\,, \quad |\beta_i|\leq C$',
'e2337': r'$f(\mathbf{x}) \;=\; \sum_{i=1}^{n}\beta_i\,k(\mathbf{x}_i,\mathbf{x}) + b\,, \qquad \mathbf{w} \;=\; \sum_{i=1}^{n}\beta_i\,\boldsymbol{\varphi}(\mathbf{x}_i)$',
'e2343': rf'$J(\mathbf{{w}},b) \;=\; \dfrac{{1}}{{2}}\left(\Vert\mathbf{{w}}\Vert^2+\dfrac{{b^2}}{{s^2}}\right) + C\sum_{{i=1}}^{{n}} L_{{\varepsilon}}\left(y_i-\mathbf{{w}}{T}\mathbf{{x}}_i-b\right)$',
'e2342': r'$\sum_{i=1}^{n}|\beta_i| \;=\; C\,\nu\,n\,, \qquad \dfrac{\#\{i:\,|r_i|>\varepsilon\}}{n}\;\leq\;\nu\;\leq\;\dfrac{n_{\mathrm{SV}}}{n}$',
'e2381': r'$\hat{f}\left(\mathbf{x};\;s\,\mathbf{y},\;C,\;s\varepsilon\right) \;=\; s\;\hat{f}\left(\mathbf{x};\;\mathbf{y},\;C/s,\;\varepsilon\right)$',
'e23101': r'$p(r) \;=\; \dfrac{C}{2\left(1+C\varepsilon\right)}\,\exp\!\left(-C\,L_{\varepsilon}(r)\right)\,, \qquad \int_{-\infty}^{\infty}p(r)\,dr=1$',
}
mkeq(E,'.')
def two(name,l1,l2,ls=2.6):
    fig=plt.figure(figsize=(.01,.01)); fig.text(0,0,l1+'\n'+l2,fontsize=11.5,linespacing=ls); fig.savefig(name+'.png',dpi=300,bbox_inches='tight',pad_inches=0.02,facecolor='white'); plt.close(fig)
def three(name,l1,l2,l3,ls=2.6):
    fig=plt.figure(figsize=(.01,.01)); fig.text(0,0,l1+'\n'+l2+'\n'+l3,fontsize=11.5,linespacing=ls); fig.savefig(name+'.png',dpi=300,bbox_inches='tight',pad_inches=0.02,facecolor='white'); plt.close(fig)
two('e2332', r'$\min_{\mathbf{w},b,\boldsymbol{\xi},\boldsymbol{\xi}^{*}}\;\; \dfrac{1}{2}\Vert\mathbf{w}\Vert^2 + C\sum_{i=1}^{n}\left(\xi_i+\xi_i^{*}\right)$',
      r'$\mathrm{subject\ to}\quad y_i-\mathbf{w}^{\mathrm{T}}\mathbf{x}_i-b \leq \varepsilon+\xi_i\,, \quad \mathbf{w}^{\mathrm{T}}\mathbf{x}_i+b-y_i \leq \varepsilon+\xi_i^{*}\,, \quad \xi_i,\;\xi_i^{*}\geq 0$')
two('e2333', r'$L \;=\; \dfrac{1}{2}\Vert\mathbf{w}\Vert^2 + C\sum_{i}\left(\xi_i+\xi_i^{*}\right) - \sum_{i}\left(\mu_i\xi_i+\mu_i^{*}\xi_i^{*}\right)$',
      r'$-\sum_{i}\alpha_i\left(\varepsilon+\xi_i-y_i+\mathbf{w}^{\mathrm{T}}\mathbf{x}_i+b\right) - \sum_{i}\alpha_i^{*}\left(\varepsilon+\xi_i^{*}+y_i-\mathbf{w}^{\mathrm{T}}\mathbf{x}_i-b\right)$')
three('e2334', r'$\dfrac{\partial L}{\partial\mathbf{w}}=\mathbf{0}:\quad \mathbf{w} \;=\; \sum_{i}\left(\alpha_i-\alpha_i^{*}\right)\mathbf{x}_i\,, \qquad \dfrac{\partial L}{\partial b}=0:\quad \sum_{i}\left(\alpha_i-\alpha_i^{*}\right)=0$',
      r'$\dfrac{\partial L}{\partial\xi_i}=0:\quad \alpha_i+\mu_i=C\,, \qquad \dfrac{\partial L}{\partial\xi_i^{*}}=0:\quad \alpha_i^{*}+\mu_i^{*}=C$',
      r'$\Longrightarrow\quad 0\leq\alpha_i\leq C\,, \qquad 0\leq\alpha_i^{*}\leq C$',ls=2.7)
two('e2335', r'$\max_{\boldsymbol{\alpha},\boldsymbol{\alpha}^{*}}\;\; -\dfrac{1}{2}\sum_{i,j}\left(\alpha_i-\alpha_i^{*}\right)\left(\alpha_j-\alpha_j^{*}\right)K_{ij} - \varepsilon\sum_{i}\left(\alpha_i+\alpha_i^{*}\right) + \sum_{i}y_i\left(\alpha_i-\alpha_i^{*}\right)$',
      r'$\mathrm{subject\ to}\quad \sum_{i}\left(\alpha_i-\alpha_i^{*}\right)=0\,, \qquad 0\leq\alpha_i,\;\alpha_i^{*}\leq C$')
two('e2338', r'$b \;=\; y_i-\varepsilon-\sum_{j}\beta_j\,k(\mathbf{x}_j,\mathbf{x}_i) \quad \mathrm{if}\;\; 0<\beta_i<C\,,$',
      r'$b \;=\; y_i+\varepsilon-\sum_{j}\beta_j\,k(\mathbf{x}_j,\mathbf{x}_i) \quad \mathrm{if}\;\; -C<\beta_i<0$')
mkeq({'e2341': r'$\min_{\mathbf{w},b,\varepsilon,\boldsymbol{\xi},\boldsymbol{\xi}^{*}}\;\; \dfrac{1}{2}\Vert\mathbf{w}\Vert^2 + C\left(\nu n\,\varepsilon+\sum_{i=1}^{n}\left(\xi_i+\xi_i^{*}\right)\right)$'},'.')
M=lambda rows:[[str(v) for v in r] for r in rows]
matimg('m2351',[r'$\begin{array}{c}\end{array}$' if False else r'',M([[2,1,0],[5,1,0],['-1',0,3]]),M([['w'],['b'],[r'\beta']]),r'$=$',M([[5],[9],[0]])])
print('eqs ok')
