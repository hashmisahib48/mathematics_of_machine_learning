# Equation and matrix images of Chapter 21. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq, matimg
import matplotlib.pyplot as plt
T=r'^{\mathrm{T}}'; w=r'\mathbf{w}'; wh=r'\hat{\mathbf{w}}'; y=r'\mathbf{y}'; Phi=r'\mathbf{\Phi}'; K=r'\mathbf{K}'; I=r'\mathbf{I}'; a=r'\mathbf{a}'
E={
'e2132': r'$\boldsymbol{\varphi}(\mathbf{x}) \;=\; \left(1,\;\sqrt{2}\,x_1,\;\sqrt{2}\,x_2,\;x_1^{2},\;\sqrt{2}\,x_1x_2,\;x_2^{2}\right)^{\mathrm{T}}$',
'e2133': r'$k(\mathbf{x},\mathbf{z}) \;=\; \boldsymbol{\varphi}(\mathbf{x})^{\mathrm{T}}\boldsymbol{\varphi}(\mathbf{z})$',
'e2135': rf'$K_{{ij}} \;=\; k(\mathbf{{x}}_i,\mathbf{{x}}_j)\,, \qquad {K} \;=\; {Phi}{Phi}{T}$',
'e2136': rf'$\mathbf{{c}}{T}{K}\,\mathbf{{c}} \;=\; \sum_{{i,j}} c_i c_j\,\boldsymbol{{\varphi}}(\mathbf{{x}}_i){T}\boldsymbol{{\varphi}}(\mathbf{{x}}_j) \;=\; \left\Vert \sum_{{i}} c_i\,\boldsymbol{{\varphi}}(\mathbf{{x}}_i)\right\Vert^2 \;\geq\; 0$',
'e2137': r'$k(\mathbf{x},\mathbf{z}) \;=\; \exp\!\left(-\gamma\,\Vert\mathbf{x}-\mathbf{z}\Vert^{2}\right)$',
'e2141': rf'${wh} \;=\; \left({Phi}{T}{Phi}+\alpha{I}\right)^{{-1}}{Phi}{T}{y}$',
'e2144': r'$f(\mathbf{x}) \;=\; \sum_{i=1}^{n} a_i\,k(\mathbf{x}_i,\mathbf{x})$',
'e2145': rf'$J({a}) \;=\; \Vert {y}-{K}{a}\Vert^2 + \alpha\,{a}{T}{K}{a}\,, \qquad \nabla J \;=\; -2{K}\left({y}-{K}{a}\right)+2\alpha{K}{a}$',
'e2146': r'$\hat{y}_i \;=\; \sum_{j=1}^{n}\dfrac{\lambda_j}{\lambda_j+\alpha}\;(\mathbf{u}_j^{\mathrm{T}}\mathbf{y})\,u_{ij}\,, \qquad \mathrm{df} \;=\; \sum_{j=1}^{n}\dfrac{\lambda_j}{\lambda_j+\alpha}$',
'e2147': rf'$\mathrm{{LOO}}_i \;=\; \dfrac{{y_i-\hat{{y}}_i}}{{1-H_{{ii}}}}\,, \qquad \mathbf{{H}} \;=\; {K}\left({K}+\alpha{I}\right)^{{-1}}$',
'e21b1': r'$k(\mathbf{x},\mathbf{z}) \;=\; \sum_{r}\varphi_r(\mathbf{x})\varphi_r(\mathbf{z}) \;\Longleftrightarrow\; \sum_{i,j}c_ic_jk(\mathbf{x}_i,\mathbf{x}_j)\geq 0 \;\; \mathrm{for\ every}\;\mathbf{c}$',
'e21b3': r'$\hat{f}(\mathbf{x}_*) \;=\; \mathbf{k}_*^{\mathrm{T}}\left(\mathbf{K}+\sigma^2\mathbf{I}\right)^{-1}\mathbf{y}\,, \qquad \mathbf{k}_*=\left(k(\mathbf{x}_1,\mathbf{x}_*),\ldots,k(\mathbf{x}_n,\mathbf{x}_*)\right)^{\mathrm{T}}$',
}
mkeq(E,'.')
def two(name,l1,l2):
    fig=plt.figure(figsize=(.01,.01)); fig.text(0,0,l1+'\n'+l2,fontsize=11.5,linespacing=2.6); fig.savefig(name+'.png',dpi=300,bbox_inches='tight',pad_inches=0.02,facecolor='white'); plt.close(fig)
two('e2134', r'$\boldsymbol{\varphi}(\mathbf{x})^{\mathrm{T}}\boldsymbol{\varphi}(\mathbf{z}) \;=\; 1 + 2x_1z_1 + 2x_2z_2 + x_1^{2}z_1^{2} + 2x_1x_2z_1z_2 + x_2^{2}z_2^{2}$',
      r'$=\; \left(1 + x_1z_1 + x_2z_2\right)^{2} \;=\; \left(\mathbf{x}^{\mathrm{T}}\mathbf{z}+1\right)^{2}$')
two('e2138', r'$e^{-\gamma(x-z)^{2}} \;=\; e^{-\gamma x^{2}}\,e^{-\gamma z^{2}}\,e^{\,2\gamma xz} \;=\; \sum_{m=0}^{\infty}\varphi_m(x)\,\varphi_m(z)$',
      r'$\varphi_m(x) \;=\; e^{-\gamma x^{2}}\sqrt{\dfrac{(2\gamma)^{m}}{m!}}\;x^{m}\,, \qquad m=0,1,2,\ldots$')
two('e2139', r'$k_1+k_2:\;\; \boldsymbol{\varphi} = \binom{\boldsymbol{\varphi}_1}{\boldsymbol{\varphi}_2}\,, \qquad k_1\,k_2:\;\; \varphi_{rs}(\mathbf{x}) = \varphi_{1r}(\mathbf{x})\,\varphi_{2s}(\mathbf{x})$',
      r'$c\,k_1\;(c\geq 0):\;\; \boldsymbol{\varphi} = \sqrt{c}\,\boldsymbol{\varphi}_1\,, \qquad k_1 + c:\;\; \mathrm{an\ extra\ feature}\;\sqrt{c}$')
two('e2143', r'$\hat{\mathbf{w}} \;=\; \boldsymbol{\Phi}^{\mathrm{T}}\left(\mathbf{K}+\alpha\mathbf{I}\right)^{-1}\mathbf{y}$',
      r'$\mathbf{a} \;=\; \left(\mathbf{K}+\alpha\mathbf{I}\right)^{-1}\mathbf{y}\,, \qquad \hat{\mathbf{w}} \;=\; \boldsymbol{\Phi}^{\mathrm{T}}\mathbf{a} \;=\; \sum_{i=1}^{n} a_i\,\boldsymbol{\varphi}(\mathbf{x}_i)$')
two('e2131', rf'$f(\mathbf{{x}}) \;=\; {w}{T}\boldsymbol{{\varphi}}(\mathbf{{x}})\,, \qquad J({w}) \;=\; \sum_{{i=1}}^{{n}}\left(y_i - {w}{T}\boldsymbol{{\varphi}}(\mathbf{{x}}_i)\right)^2 + \alpha\,\Vert {w}\Vert^2$',
      rf'$=\; \Vert {y}-{Phi}{w}\Vert^2 + \alpha\,\Vert {w}\Vert^2$')
two('e2142', rf'$\left({Phi}{T}{Phi}+\alpha{I}\right){Phi}{T} \;=\; {Phi}{T}\left({Phi}{Phi}{T}+\alpha{I}\right)$',
      rf'$\Longrightarrow\quad \left({Phi}{T}{Phi}+\alpha{I}\right)^{{-1}}{Phi}{T} \;=\; {Phi}{T}\left({Phi}{Phi}{T}+\alpha{I}\right)^{{-1}}$')
two('e21b2', r'$\hat{f} \;=\; \arg\min_{f}\; \sum_{i=1}^{n}\left(y_i-f(\mathbf{x}_i)\right)^2 + \alpha\,\Vert f\Vert^2_{\mathcal{H}}$',
      r'$\Longrightarrow\quad \hat{f}(\mathbf{x}) \;=\; \sum_{i=1}^{n} a_i\,k(\mathbf{x}_i,\mathbf{x})$')
M=lambda rows:[[str(v) for v in r] for r in rows]
matimg('m2151',[r'$\mathbf{K}+0.1\,\mathbf{I}=$',M([['1.1',0,-1],[0,'0.1',0],[-1,0,'1.1']]),r'$,\quad \mathbf{y}=$',M([[1],[0],[1]])])
matimg('m2152',[r'$\mathbf{K}=$',M([['1','0.60653','0.13534'],['0.60653','1','0.60653'],['0.13534','0.60653','1']])])
two('e2153', r'$1.23534\,a_1 + 0.60653\,a_2 \;=\; 1\,, \qquad\quad 1.21306\,a_1 + 1.1\,a_2 \;=\; 0$',
      r'$a_2 \;=\; -1.10278\,a_1\,, \qquad 0.56646\,a_1 \;=\; 1$')
two('e2154', r'$f(0.5) \;=\; 0.32465\,a_1 + 0.88250\,a_2 + 0.88250\,a_3$',
      r'$=\; 0.5731 - 1.7180 + 1.5579 \;=\; 0.4130$')
print('eqs ok')
