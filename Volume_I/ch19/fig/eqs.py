# Equation images of Chapter 19. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq
import matplotlib.pyplot as plt
AM=r'\mathrm{arg\,max}'
E={
'e1931': r'$P(k\mid\mathbf{x}) \;=\; \dfrac{P(k)\;p(\mathbf{x}\mid k)}{\sum_{l=1}^{K} P(l)\;p(\mathbf{x}\mid l)}$',
'e1932': r'$p(\mathbf{x}\mid k) \;=\; p(x_1\mid k)\,p(x_2\mid k)\cdots p(x_p\mid k) \;=\; \prod_{j=1}^{p} p(x_j\mid k)$',
'e1933': rf'$s_k \;=\; \ln P(k) + \sum_{{j=1}}^{{p}} \ln p(x_j\mid k)\,, \qquad \hat{{y}} \;=\; {AM}_k\; s_k$',
'e1934': r'$P(k\mid\mathbf{x}) \;=\; \dfrac{e^{\,s_k}}{\sum_{l=1}^{K} e^{\,s_l}}$',
'e1935': r'$\ln\sum_{l} e^{\,s_l} \;=\; m + \ln\sum_{l} e^{\,s_l-m}\,, \qquad m=\max_l s_l$',
'e1936': r'$P(k) \;=\; \dfrac{n_k}{n}\,, \qquad \ln P(k) \;=\; \ln n_k - \ln n$',
'e1938': r'$\theta_j \;=\; \dfrac{N_j}{N}\,, \qquad N=\sum_{j} N_j$',
'e1941': r'$p(x_j\mid k) \;=\; \dfrac{1}{\sqrt{2\pi\sigma_{kj}^{2}}}\;\exp\!\left(-\dfrac{(x_j-\mu_{kj})^{2}}{2\sigma_{kj}^{2}}\right)$',
'e1943': r'$s_k \;=\; \ln P(k) \;-\; \dfrac{1}{2}\sum_{j=1}^{p}\left[\ln\left(2\pi\sigma_{kj}^{2}\right) + \dfrac{(x_j-\mu_{kj})^{2}}{\sigma_{kj}^{2}}\right]$',
'e1944': r'$p(\mathbf{x}\mid k) \;=\; \dfrac{m!}{x_1!\cdots x_p!}\;\prod_{j=1}^{p}\theta_{kj}^{\,x_j}\,, \qquad m=\sum_j x_j$',
'e1945': r'$\theta_{kj} \;=\; \dfrac{N_{kj}+\alpha}{N_k+\alpha p}\,, \qquad N_{kj}=\sum_{i\in k} x_{ij}\,, \quad N_k=\sum_{j} N_{kj}$',
'e1946': r'$s_k \;=\; \ln P(k) + \sum_{j=1}^{p} x_j\,\ln\theta_{kj}$',
'e1947': rf'$\tilde{{\theta}}_{{kj}} \;=\; \dfrac{{\sum_{{l\neq k}} N_{{lj}}+\alpha}}{{\sum_{{l\neq k}} N_{{l}}+\alpha p}}\,, \qquad s_k \;=\; -\sum_{{j=1}}^{{p}} x_j\,\ln\tilde{{\theta}}_{{kj}}\,, \qquad \hat{{y}}={AM}_k\, s_k$',
}
mkeq(E,'.')
import matplotlib
def two(name,l1,l2):
    fig=plt.figure(figsize=(.01,.01)); fig.text(0,0,l1+'\n'+l2,fontsize=11.5,linespacing=2.6); fig.savefig(name+'.png',dpi=300,bbox_inches='tight',pad_inches=0.02,facecolor='white'); plt.close(fig)
two('e1937',r'$\mathcal{L}(\boldsymbol{\theta},\lambda) \;=\; \sum_j N_j\ln\theta_j-\lambda\left(\sum_j\theta_j-1\right)$',r'$\dfrac{\partial\mathcal{L}}{\partial\theta_j}=\dfrac{N_j}{\theta_j}-\lambda=0 \;\;\Rightarrow\;\; \theta_j=\dfrac{N_j}{\lambda}$')
two('e1942',r'$\mu_{kj} \;=\; \dfrac{1}{n_k}\sum_{i\in k} x_{ij}$',r'$\sigma_{kj}^{2} \;=\; \dfrac{1}{n_k}\sum_{i\in k}\,(x_{ij}-\mu_{kj})^{2} + \varepsilon\,, \qquad \varepsilon=\texttt{var\_smoothing}\times\max_j \mathrm{Var}(x_j)$'.replace(r'\texttt{var\_smoothing}',r'v'))
two('e1948',r'$\theta_{kj} \;=\; \dfrac{N_{kj}+\alpha}{n_k+2\alpha}\,, \qquad N_{kj}=\#\{i\in k:\ x_{ij}=1\}$',r'$s_k \;=\; \ln P(k) + \sum_{j=1}^{p}\left[\,x_j\ln\theta_{kj} + (1-x_j)\ln(1-\theta_{kj})\right]$')
two('e1949',r'$\theta_{kjc} \;=\; \dfrac{N_{kjc}+\alpha}{n_k+\alpha\,m_j}\,, \qquad N_{kjc}=\#\{i\in k:\ x_{ij}=c\}$',r'$s_k \;=\; \ln P(k) + \sum_{j=1}^{p}\ln\theta_{k\,j\,x_j}$')
two('e19410',r'$\ln\dfrac{P(1\mid\mathbf{x})}{P(0\mid\mathbf{x})} \;=\; b + \sum_{j=1}^{p} w_j\,x_j$',r'$w_j=\ln\theta_{1j}-\ln\theta_{0j}\,, \qquad b=\ln P(1)-\ln P(0)$')
two('e19411',r'$w_j \;=\; \ln\dfrac{\theta_{1j}\,(1-\theta_{0j})}{\theta_{0j}\,(1-\theta_{1j})}$',r'$b \;=\; \ln\dfrac{P(1)}{P(0)} + \sum_{j=1}^{p}\ln\dfrac{1-\theta_{1j}}{1-\theta_{0j}}$')
two('e19101',r'$\boldsymbol{\theta}_k\mid\text{counts} \;\sim\; \mathrm{Dirichlet}\,(N_{k1}+\alpha,\ \ldots,\ N_{kp}+\alpha)$',r'$\mathrm{E}\,[\theta_{kj}\mid\text{counts}] \;=\; \dfrac{N_{kj}+\alpha}{N_k+\alpha p}$')
print('eqs ok')
