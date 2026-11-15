# Equation images of Chapter 20. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq
import matplotlib.pyplot as plt
T=r'^{\mathrm{T}}'
E={
'e2031': r'$d_q(\mathbf{x},\mathbf{z}) \;=\; \left(\sum_{j=1}^{p}\,\left|x_j - z_j\right|^{q}\right)^{1/q}$',
'e2032': r'$d_\infty(\mathbf{x},\mathbf{z}) \;=\; \lim_{q\to\infty} d_q(\mathbf{x},\mathbf{z}) \;=\; \max_{j}\,\left|x_j - z_j\right|$',
'e2033': rf'$d_{{\cos}}(\mathbf{{x}},\mathbf{{z}}) \;=\; 1-\dfrac{{\mathbf{{x}}{T}\mathbf{{z}}}}{{\Vert\mathbf{{x}}\Vert\,\Vert\mathbf{{z}}\Vert}}$',
'e2034': r'$v_c(\mathbf{x}) \;=\; \sum_{i\in N_k(\mathbf{x})} w_i\,[\,y_i = c\,]$',
'e2035': r'$w_i \;=\; 1 \quad\mathrm{(uniform)}\,, \qquad\qquad w_i \;=\; \dfrac{1}{d_i} \quad\mathrm{(distance)}$',
'e2036': r'$\hat{P}(c\,|\,\mathbf{x}) \;=\; \dfrac{v_c(\mathbf{x})}{\sum_{c^\prime} v_{c^\prime}(\mathbf{x})}\,, \qquad\qquad \hat{y} \;=\; \underset{c}{\mathrm{arg\,max}}\;\, v_c(\mathbf{x})$',
'e2037': r'$\hat{y}(\mathbf{x}) \;=\; \dfrac{\sum_{i\in N_k(\mathbf{x})} w_i\,y_i}{\sum_{i\in N_k(\mathbf{x})} w_i}$',
'e2041': r'$\mathrm{share\ of\ the\ ball\ within\ distance}\ \varepsilon\ \mathrm{of\ its\ surface} \;=\; 1-(1-\varepsilon)^{d}$',
'e2042': r'$\mathrm{edge\ of\ the\ cube\ holding\ a\ share}\ f\ \mathrm{of\ uniform\ data} \;=\; f^{\,1/d}$',
'e2051': r'$d_{\min}(\mathbf{x}_0,B) \;=\; \left(\sum_{j=1}^{p}\,\max\left(l_j - x_{0j},\; 0,\; x_{0j}-u_j\right)^{q}\right)^{1/q}$',
'e2061': rf'$\boldsymbol{{\mu}}_k \;=\; \dfrac{{1}}{{n_k}}\sum_{{i:\,y_i=k}}\mathbf{{x}}_i\,, \qquad\qquad \hat{{y}} \;=\; \underset{{k}}{{\mathrm{{arg\,min}}}}\;\Vert \mathbf{{x}}-\boldsymbol{{\mu}}_k\Vert$',
'e2063': r'$p_{ij} \;=\; \dfrac{\exp\left(-\Vert \mathbf{A}\mathbf{x}_i-\mathbf{A}\mathbf{x}_j\Vert^2\right)}{\sum_{l\neq i}\exp\left(-\Vert \mathbf{A}\mathbf{x}_i-\mathbf{A}\mathbf{x}_l\Vert^2\right)}\,, \qquad p_{ii}=0$',
'e2064': r'$f(\mathbf{A}) \;=\; \sum_{i=1}^{n} p_i\,, \qquad\qquad p_i \;=\; \sum_{j:\,y_j=y_i} p_{ij}$',
'e2065': rf'$\dfrac{{\partial f}}{{\partial \mathbf{{A}}}} \;=\; 2\mathbf{{A}}\sum_{{i}}\left(p_i\sum_{{l}}p_{{il}}\,\mathbf{{x}}_{{il}}\mathbf{{x}}_{{il}}{T}\;-\;\sum_{{j:\,y_j=y_i}}p_{{ij}}\,\mathbf{{x}}_{{ij}}\mathbf{{x}}_{{ij}}{T}\right)$',
'e20121': r'$R^{*} \;\leq\; R_{\mathrm{NN}} \;\leq\; 2R^{*}\,(1-R^{*}) \;\leq\; 2R^{*}$',
'e20122': r'$R_{\mathrm{NN}} \;=\; \mathbb{E}\left[\,2\,\eta(\mathbf{x})\,\left(1-\eta(\mathbf{x})\right)\right], \qquad \eta(\mathbf{x}) \;=\; P(y=1\,|\,\mathbf{x})$',
}
mkeq(E,'.')
def ml(name,lines,ls=2.7):
    fig=plt.figure(figsize=(.01,.01)); fig.text(0,0,'\n'.join(lines),fontsize=11.5,linespacing=ls); fig.savefig(name+'.png',dpi=300,bbox_inches='tight',pad_inches=0.02,facecolor='white'); plt.close(fig)
ml('e2062',[r'$m_k \;=\; \sqrt{\dfrac{1}{n_k}-\dfrac{1}{n}}\,, \qquad d_{kj} \;=\; \dfrac{\mu_{kj}-\bar{x}_j}{m_k\,(s_j+s_0)}$',
            r'$d_{kj}^{\prime} \;=\; \mathrm{sign}(d_{kj})\,\max\left(\left|d_{kj}\right|-\Delta,\; 0\right), \qquad \mu_{kj}^{\prime} \;=\; \bar{x}_j + m_k\,(s_j+s_0)\,d_{kj}^{\prime}$'])
print('eqs ok')
