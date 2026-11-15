# Equation and matrix images of Chapter 22. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq, matimg
import matplotlib.pyplot as plt
T=r'^{\mathrm{T}}'
def multi(name,lines,ls=2.6):
    fig=plt.figure(figsize=(.01,.01)); fig.text(0,0,'\n'.join(lines),fontsize=11.5,linespacing=ls); fig.savefig(name+'.png',dpi=300,bbox_inches='tight',pad_inches=0.02,facecolor='white'); plt.close(fig)
S={ # single-line equations
'e2231': r'$f(\mathbf{x}) \;=\; \mathbf{w}^{\mathrm{T}}\mathbf{x}+b\,, \qquad \hat{y} \;=\; \mathrm{sign}\,f(\mathbf{x})\,, \qquad y\in\{-1,+1\}$',
'e2232': r'$d_i \;=\; \dfrac{y_i\,f(\mathbf{x}_i)}{\Vert\mathbf{w}\Vert} \;=\; \dfrac{y_i\,(\mathbf{w}^{\mathrm{T}}\mathbf{x}_i+b)}{\Vert\mathbf{w}\Vert}$',
'e2234': r'$\min_{\mathbf{w},\,b}\ \dfrac{1}{2}\,\Vert\mathbf{w}\Vert^{2} \qquad \mathrm{subject\ to} \qquad y_i\,(\mathbf{w}^{\mathrm{T}}\mathbf{x}_i+b)-1\;\geq\;0\,, \quad i=1,\ldots,n$',
'e2235': r'$L(\mathbf{w},b,\boldsymbol{\alpha}) \;=\; \dfrac{1}{2}\Vert\mathbf{w}\Vert^{2}-\sum_{i=1}^{n}\alpha_i\left[\,y_i\,(\mathbf{w}^{\mathrm{T}}\mathbf{x}_i+b)-1\,\right]\,, \qquad \alpha_i\geq 0$',
'e2238': r'$\alpha_i\left[\,y_i\,f(\mathbf{x}_i)-1\,\right] \;=\; 0\,, \qquad b \;=\; y_k-\sum_{i=1}^{n}\alpha_i\,y_i\,\mathbf{x}_i^{\mathrm{T}}\mathbf{x}_k \quad (\alpha_k>0)$',
'e2239': r'$W(\boldsymbol{\alpha}) \;=\; \mathbf{1}^{\mathrm{T}}\boldsymbol{\alpha}-\dfrac{1}{2}\,\boldsymbol{\alpha}^{\mathrm{T}}\mathbf{Q}\,\boldsymbol{\alpha}\,, \qquad Q_{ij}=y_iy_j\,\mathbf{x}_i^{\mathrm{T}}\mathbf{x}_j\,, \qquad \Vert\mathbf{w}\Vert^{2}=\boldsymbol{\alpha}^{\mathrm{T}}\mathbf{Q}\boldsymbol{\alpha}=\sum_i\alpha_i$',
'e2241': r'$y_i\,f(\mathbf{x}_i) \;\geq\; 1-\xi_i\,, \qquad \xi_i\;\geq\; 0\,, \qquad i=1,\ldots,n$',
'e2242': r'$\min_{\mathbf{w},\,b,\,\boldsymbol{\xi}}\ \dfrac{1}{2}\,\Vert\mathbf{w}\Vert^{2}+C\sum_{i=1}^{n}\xi_i \qquad \mathrm{subject\ to\ Eq.\ 22.4\text{-}1}$',
'e2243': r'$J(\mathbf{w},b) \;=\; \dfrac{1}{2}\,\Vert\mathbf{w}\Vert^{2}+C\sum_{i=1}^{n}\max\left(0,\;1-y_i\,f(\mathbf{x}_i)\right)$',
'e2244': r'$\dfrac{J}{Cn} \;=\; \dfrac{1}{n}\sum_{i=1}^{n}\max\left(0,\,1-y_i f(\mathbf{x}_i)\right)+\alpha_{\mathrm{SGD}}\cdot\dfrac{1}{2}\Vert\mathbf{w}\Vert^{2}\,, \qquad \alpha_{\mathrm{SGD}} \;=\; \dfrac{1}{Cn}$',
'e2247': r'$\alpha_i=0\;\Rightarrow\; y_if(\mathbf{x}_i)\geq 1\,, \qquad 0<\alpha_i<C\;\Rightarrow\; y_if(\mathbf{x}_i)=1\,, \qquad \alpha_i=C\;\Rightarrow\; y_if(\mathbf{x}_i)\leq 1$',
'e2248': r'$b \;=\; y_k-\sum_{i=1}^{n}\alpha_i\,y_i\,k(\mathbf{x}_i,\mathbf{x}_k) \qquad \mathrm{for\ any}\ k\ \mathrm{with}\ 0<\alpha_k<C$',
'e2249': r'$0\;\leq\;\alpha_i\;\leq\;C_i \;=\; C\cdot\mathrm{class\_weight}[y_i]\cdot\mathrm{sample\_weight}_i$',
'e2251': r'$f(\mathbf{x}) \;=\; \sum_{i\in\mathrm{SV}}\alpha_i\,y_i\,k(\mathbf{x}_i,\mathbf{x})+b$',
'e2252': r'$K_{ij} \;=\; k(\mathbf{x}_i,\mathbf{x}_j) \;=\; \varphi(\mathbf{x}_i)^{\mathrm{T}}\varphi(\mathbf{x}_j)\,, \qquad Q_{ij}=y_iy_jK_{ij}\,, \qquad \Vert\mathbf{w}\Vert^{2}=\boldsymbol{\alpha}^{\mathrm{T}}\mathbf{Q}\boldsymbol{\alpha}$',
'e2253': r'$k(\mathbf{x},\mathbf{z}) \;=\; \exp\left(-\gamma\,\Vert\mathbf{x}-\mathbf{z}\Vert^{2}\right)$',
'e2254': r'$\gamma_{\mathrm{scale}} \;=\; \dfrac{1}{p\cdot\mathrm{Var}(\mathbf{X})}\,, \qquad \gamma_{\mathrm{auto}} \;=\; \dfrac{1}{p}\,, \qquad \mathrm{Var}(\mathbf{X})=\dfrac{1}{np}\sum_{i,j}\left(x_{ij}-\bar{x}\right)^{2}$',
'e2261': r'$y_i\,\alpha_i+y_j\,\alpha_j \;=\; c \quad (\mathrm{kept\ fixed}), \qquad \alpha_i \;=\; y_i c-y_iy_j\,\alpha_j$',
'e2263': r'$\alpha_j^{\mathrm{new}} \;=\; \alpha_j+\dfrac{y_j\,(E_i-E_j)}{\eta}\,, \qquad E_k=f(\mathbf{x}_k)-y_k\,, \qquad \eta=K_{ii}+K_{jj}-2K_{ij}$',
'e2264': r'$\alpha_i^{\mathrm{new}} \;=\; \alpha_i+y_iy_j\left(\alpha_j-\alpha_j^{\mathrm{new}}\right)$',
'e22101': r'$J(\mathbf{w},b) \;=\; \dfrac{1}{2}\Vert\mathbf{w}\Vert^{2}+\dfrac{b^{2}}{2c^{2}}+C\sum_{i=1}^{n}\max\left(0,\,1-y_i f(\mathbf{x}_i)\right)^{2}$',
}
M={ # multi-line equations
'e2233':[r'$y_i\,(\mathbf{w}^{\mathrm{T}}\mathbf{x}_i+b)\;\geq\;1\,, \quad i=1,\ldots,n$', r'$\mathrm{width\ of\ the\ street} \;=\; \dfrac{2}{\Vert\mathbf{w}\Vert}$'],
'e2236':[r'$\dfrac{\partial L}{\partial\mathbf{w}} \;=\; \mathbf{w}-\sum_i\alpha_i\,y_i\,\mathbf{x}_i \;=\; \mathbf{0} \quad\Rightarrow\quad \mathbf{w} \;=\; \sum_{i=1}^{n}\alpha_i\,y_i\,\mathbf{x}_i$', r'$\dfrac{\partial L}{\partial b} \;=\; -\sum_i\alpha_i\,y_i \;=\; 0 \quad\Rightarrow\quad \sum_{i=1}^{n}\alpha_i\,y_i \;=\; 0$'],
'e2237':[r'$\max_{\boldsymbol{\alpha}}\ W(\boldsymbol{\alpha}) \;=\; \sum_{i=1}^{n}\alpha_i-\dfrac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_i\alpha_j\,y_iy_j\,\mathbf{x}_i^{\mathrm{T}}\mathbf{x}_j$', r'$\mathrm{subject\ to}\quad \alpha_i\geq 0\,, \qquad \sum_{i=1}^{n}\alpha_i\,y_i \;=\; 0$'],
'e2245':[r'$\dfrac{\partial L}{\partial\xi_i} \;=\; C-\alpha_i-\mu_i \;=\; 0\,, \qquad \alpha_i\geq 0\,, \quad \mu_i\geq 0 \quad\Rightarrow\quad 0\;\leq\;\alpha_i\;\leq\;C$'],
'e2246':[r'$\max_{\boldsymbol{\alpha}}\ W(\boldsymbol{\alpha}) \;=\; \sum_{i=1}^{n}\alpha_i-\dfrac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_i\alpha_j\,y_iy_j\,K_{ij}$', r'$\mathrm{subject\ to}\quad 0\;\leq\;\alpha_i\;\leq\;C\,, \qquad \sum_{i=1}^{n}\alpha_i\,y_i \;=\; 0$'],
'e2262':[r'$\dfrac{\mathrm{d}W}{\mathrm{d}\alpha_j} \;=\; y_j\,(E_i-E_j)\,, \qquad \dfrac{\mathrm{d}^{2}W}{\mathrm{d}\alpha_j^{2}} \;=\; -\,\eta \;\leq\; 0$'],
'e2265':[r'$y_i\neq y_j:\quad L=\max(0,\;\alpha_j-\alpha_i)\,, \quad H=\min(C,\;C+\alpha_j-\alpha_i)$', r'$y_i=y_j:\quad L=\max(0,\;\alpha_i+\alpha_j-C)\,, \quad H=\min(C,\;\alpha_i+\alpha_j)$'],
'e2266':[r'$v_k \;=\; y_k-\sum_{m}\alpha_m\,y_m\,K_{mk}$', r'$i\;=\;\arg\max_{k\in I_{\mathrm{up}}}v_k\,, \qquad j\;=\;\arg\max_{t\in I_{\mathrm{low}},\ v_t<v_i}\dfrac{(v_i-v_t)^{2}}{\eta_{it}}$', r'$\mathrm{stop\ when}\quad \max_{I_{\mathrm{up}}}v-\min_{I_{\mathrm{low}}}v \;<\; \mathrm{tol}\,, \qquad b \;=\; \dfrac{1}{2}\left(\max_{I_{\mathrm{up}}}v+\min_{I_{\mathrm{low}}}v\right)$'],
'e2271':[r'$-25\alpha_3+40\alpha_7+30\alpha_8+b \;=\; -1 \qquad (k=3)$', r'$-40\alpha_3+65\alpha_7+45\alpha_8+b \;=\; +1 \qquad (k=7)$', r'$-30\alpha_3+45\alpha_7+45\alpha_8+b \;=\; +1 \qquad (k=8)$', r'$-\alpha_3+\alpha_7+\alpha_8 \;=\; 0$'],
}
mkeq(S,'.')
for k,v in M.items(): multi(k,v,2.0 if k=='e2271' else 2.6)
print('eqs ok')
