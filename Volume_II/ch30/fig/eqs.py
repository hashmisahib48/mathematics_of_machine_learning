# Equation and matrix images of Chapter 30. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq, matimg
import matplotlib.pyplot as plt
T=r'^{\mathrm{T}}'; y=r'\mathbf{y}'; S=r'\mathbf{S}'; K=r'\mathbf{K}'; I=r'\mathbf{I}'; ks=r'\mathbf{k}_*'; a=r'\mathbf{a}'; x_=r'\mathbf{x}'
Si=r'\mathbf{S}^{-1}'
def two(name,*ls,sp=2.6):
    fig=plt.figure(figsize=(.01,.01)); fig.text(0,0,'\n'.join(ls),fontsize=11.5,linespacing=sp); fig.savefig(name+'.png',dpi=300,bbox_inches='tight',pad_inches=0.02,facecolor='white'); plt.close(fig)
E={
'e3039': rf'$m(\mathbf{{x}}) \;=\; \sum_{{i=1}}^{{n}} a_i\,k(\mathbf{{x}}_i,\mathbf{{x}})\,, \qquad {a} \;=\; {Si}{y}$',
'e3043': rf'$\dfrac{{\partial}}{{\partial\theta_j}}\,\ln p({y}\,|\,\mathbf{{X}},\theta) \;=\; \dfrac{{1}}{{2}}\,\mathrm{{tr}}\!\left[\left({a}{a}{T}-{Si}\right)\dfrac{{\partial {S}}}{{\partial\theta_j}}\right]$',
'e3051': r'$k(\mathbf{x},\mathbf{x}\,^{\prime}) \;=\; \sigma_f^{2}\,\exp\!\left(-\dfrac{\Vert\mathbf{x}-\mathbf{x}\,^{\prime}\Vert^{2}}{2\,\ell^{2}}\right)\,, \qquad \gamma \;=\; \dfrac{1}{2\ell^{2}}$',
'e3052': r'$k(d) \;=\; \dfrac{2^{1-\nu}}{\Gamma(\nu)}\left(\sqrt{2\nu}\,\dfrac{d}{\ell}\right)^{\nu} K_{\nu}\!\left(\sqrt{2\nu}\,\dfrac{d}{\ell}\right)\,, \qquad d=\Vert\mathbf{x}-\mathbf{x}\,^{\prime}\Vert$',
'e30105': rf'$\hat{{\mathbf{{f}}}} \;=\; {K}\,({y}-\hat{{\boldsymbol{{\pi}}}})\,, \qquad p(\mathbf{{f}}\,|\,{y}) \;\approx\; \mathcal{{N}}\!\left(\hat{{\mathbf{{f}}}},\,\left({K}^{{-1}}+\hat{{\mathbf{{W}}}}\right)^{{-1}}\right)$',
'e30106': rf'$\bar{{f}}_* \;=\; {ks}{T}\left({y}-\hat{{\boldsymbol{{\pi}}}}\right)$',
'e30107': rf'$v_* \;=\; k_{{**}} - {ks}{T}\left({K}+\hat{{\mathbf{{W}}}}^{{-1}}\right)^{{-1}}{ks}$',
'e30108': r'$P(y_*=1\,|\,\mathbf{y}) \;\approx\; \int \dfrac{1}{1+e^{-z}}\;\mathcal{N}(z\,;\,\bar{f}_*,\,v_*)\,dz$',
'e30109': rf'$\ln q({y}\,|\,\mathbf{{X}},\theta) \;=\; -\dfrac{{1}}{{2}}\hat{{\mathbf{{f}}}}{T}{K}^{{-1}}\hat{{\mathbf{{f}}}} + \sum_i \ln\sigma\left((2y_i-1)\hat{{f}}_i\right) - \dfrac{{1}}{{2}}\ln\left|\mathbf{{I}}+\hat{{\mathbf{{W}}}}^{{1/2}}{K}\hat{{\mathbf{{W}}}}^{{1/2}}\right|$',
}
mkeq(E,'.')
two('e3031', r'$\mathbf{f} \;=\; \left(f(\mathbf{x}_1),\,\ldots,\,f(\mathbf{x}_m)\right)^{\mathrm{T}} \;\sim\; \mathcal{N}\left(\mathbf{0},\,\mathbf{K}\right)$',
     r'$K_{ij} \;=\; k(\mathbf{x}_i,\mathbf{x}_j)\,, \qquad \mathrm{written}\;\; f \;\sim\; \mathcal{GP}(0,\,k)$')
two('e3032', r'$f(\mathbf{x}) \;=\; \mathbf{w}^{\mathrm{T}}\boldsymbol{\varphi}(\mathbf{x})\,, \qquad \mathbf{w} \;\sim\; \mathcal{N}(\mathbf{0},\,\mathbf{I})$',
     r'$\mathrm{Cov}\left(f(\mathbf{x}),\,f(\mathbf{x}\,^{\prime})\right) \;=\; \boldsymbol{\varphi}(\mathbf{x})^{\mathrm{T}}\boldsymbol{\varphi}(\mathbf{x}\,^{\prime}) \;=\; k(\mathbf{x},\,\mathbf{x}\,^{\prime})$')
two('e3033', rf'$y_i \;=\; f(\mathbf{{x}}_i)+\varepsilon_i\,, \qquad \varepsilon_i \;\sim\; \mathcal{{N}}(0,\,\sigma^2)$',
     rf'$\Longrightarrow\quad {y} \;\sim\; \mathcal{{N}}\left(\mathbf{{0}},\,{S}\right)\,, \qquad {S} \;=\; {K}+\sigma^2{I}$')
two('e3035', rf'$z \;=\; f_* - {ks}{T}{Si}{y}$',
     rf'$\mathrm{{Cov}}(z,\,{y}) \;=\; {ks}{T} - {ks}{T}{Si}{S} \;=\; \mathbf{{0}}{T}$')
two('e3036', rf'$\mathrm{{Var}}(z) \;=\; k_{{**}} - 2\,{ks}{T}{Si}{ks} + {ks}{T}{Si}{S}{Si}{ks}$',
     rf'$=\; k_{{**}} - {ks}{T}{Si}{ks}$')
two('e3037', rf'$f_*\,|\,{y} \;\sim\; \mathcal{{N}}\left(m(\mathbf{{x}}_*),\,v(\mathbf{{x}}_*)\right)$',
     rf'$m(\mathbf{{x}}_*) \;=\; {ks}{T}{Si}{y}\,, \qquad v(\mathbf{{x}}_*) \;=\; k_{{**}} - {ks}{T}{Si}{ks}$')
two('e3038', r'$\mathbf{m} \;=\; \mathbf{K}_*^{\mathrm{T}}\,\mathbf{S}^{-1}\mathbf{y}\,, \qquad \mathbf{V} \;=\; \mathbf{K}_{**} - \mathbf{K}_*^{\mathrm{T}}\,\mathbf{S}^{-1}\mathbf{K}_*$',
     r'$(\mathbf{K}_*)_{ij}=k(\mathbf{x}_i,\mathbf{x}_{*j})\,, \qquad (\mathbf{K}_{**})_{jl}=k(\mathbf{x}_{*j},\mathbf{x}_{*l})$')
two('e3041', rf'${S} \;=\; \mathbf{{L}}\mathbf{{L}}{T}\,, \qquad {a} \;=\; \mathbf{{L}}{T}\backslash\left(\mathbf{{L}}\backslash{y}\right)\,, \qquad m(\mathbf{{x}}_*) \;=\; {ks}{T}{a}$',
     rf'$\mathbf{{v}} \;=\; \mathbf{{L}}\backslash{ks}\,, \qquad v(\mathbf{{x}}_*) \;=\; k_{{**}} - \mathbf{{v}}{T}\mathbf{{v}}$')
two('e3042', rf'$\ln p({y}\,|\,\mathbf{{X}},\theta) \;=\; -\dfrac{{1}}{{2}}\,{y}{T}{Si}{y} \;-\; \dfrac{{1}}{{2}}\ln|{S}| \;-\; \dfrac{{n}}{{2}}\ln 2\pi$',
     rf'$=\; -\dfrac{{1}}{{2}}\,{y}{T}{a} \;-\; \sum_{{i=1}}^{{n}}\ln L_{{ii}} \;-\; \dfrac{{n}}{{2}}\ln 2\pi$',sp=2.9)
two('e30101', r'$P(y_i=1\,|\,f_i) \;=\; \sigma(f_i) \;=\; \dfrac{1}{1+e^{-f_i}}\,, \qquad \mathbf{f} \;\sim\; \mathcal{N}(\mathbf{0},\,\mathbf{K})$',
     r'$\mathrm{labels}\;\; y_i\in\{0,1\}\,, \quad \mathrm{independent\ given}\;\mathbf{f}$')
two('e30102', rf'$\Psi(\mathbf{{f}}) \;=\; \ln p({y}\,|\,\mathbf{{f}}) + \ln p(\mathbf{{f}}) \;=\; \sum_i \ln p(y_i\,|\,f_i) \;-\; \dfrac{{1}}{{2}}\mathbf{{f}}{T}{K}^{{-1}}\mathbf{{f}} \;+\; \mathrm{{const}}$',
     r'$\pi_i=\sigma(f_i)$',sp=2.9)
two('e30103', rf'$\nabla\Psi(\mathbf{{f}}) \;=\; ({y}-\boldsymbol{{\pi}}) - {K}^{{-1}}\mathbf{{f}}\,, \qquad \nabla^2\Psi(\mathbf{{f}}) \;=\; -\mathbf{{W}} - {K}^{{-1}}$',
      r'$\mathbf{W} \;=\; \mathrm{diag}\left(\pi_1(1-\pi_1),\,\ldots,\,\pi_n(1-\pi_n)\right)$')
two('e30104', rf'$\mathbf{{f}}_{{\mathrm{{new}}}} \;=\; \left({K}^{{-1}}+\mathbf{{W}}\right)^{{-1}}\left(\mathbf{{W}}\mathbf{{f}} + {y}-\boldsymbol{{\pi}}\right)$',
      rf'$\mathbf{{b}}=\mathbf{{W}}\mathbf{{f}}+{y}-\boldsymbol{{\pi}}\,, \qquad \mathbf{{B}}=\mathbf{{I}}+\mathbf{{W}}^{{1/2}}{K}\mathbf{{W}}^{{1/2}}$',
      rf'${a}=\mathbf{{b}}-\mathbf{{W}}^{{1/2}}\mathbf{{B}}^{{-1}}\mathbf{{W}}^{{1/2}}{K}\mathbf{{b}}\,, \qquad \mathbf{{f}}_{{\mathrm{{new}}}}={K}{a}$')
two('e30121', r'$\sigma^2\,\boldsymbol{\varphi}_*^{\mathrm{T}}\left(\boldsymbol{\Phi}^{\mathrm{T}}\boldsymbol{\Phi}+\sigma^2\mathbf{I}\right)^{-1}\boldsymbol{\varphi}_* \;=\; \boldsymbol{\varphi}_*^{\mathrm{T}}\boldsymbol{\varphi}_*-\boldsymbol{\varphi}_*^{\mathrm{T}}\boldsymbol{\Phi}^{\mathrm{T}}\left(\boldsymbol{\Phi}\boldsymbol{\Phi}^{\mathrm{T}}+\sigma^2\mathbf{I}\right)^{-1}\boldsymbol{\Phi}\boldsymbol{\varphi}_*$',
      r'$=\; k_{**}-\mathbf{k}_*^{\mathrm{T}}\mathbf{S}^{-1}\mathbf{k}_*$')
M=lambda rows:[[str(v) for v in r] for r in rows]
matimg('m3034',[M([[r'\mathbf{y}'],[r'f_*']]),r'$\sim\;\mathcal{N}(\mathbf{0},$',M([[r'\mathbf{S}',r'\mathbf{k}_*'],[r'\mathbf{k}_*^{\mathrm{T}}',r'k_{**}']]),r'$)$'],rh=0.26)
matimg('m3061',[r'$\mathbf{S}=$',M([['1.1','0.13534'],['0.13534','1.1']]),r'$,\quad \mathbf{y}=$',M([['1.2'],['0.4']])])
matimg('m3062',[r'$\mathbf{S}^{-1}=\frac{1}{1.19168}$',M([['1.1','-0.13534'],['-0.13534','1.1']]),r'$,\quad \mathbf{a}=\mathbf{S}^{-1}\mathbf{y}=$',M([['1.06222'],['0.23296']])])
two('e3063', r'$\mathbf{k}_* \;=\; \left(e^{-1/2},\,e^{-1/2}\right) \;=\; (0.60653,\;0.60653)$',
     r'$m \;=\; 0.60653\times(1.06222+0.23296) \;=\; 0.7856$',
     r'$v \;=\; 1 - 0.36788\times\dfrac{1.1+1.1-2\times 0.13534}{1.19168} \;=\; 0.4044$')
two('e3064', r'$\mathbf{k}_* \;=\; \left(e^{-8},\,e^{-2}\right) \;=\; (0.00034,\;0.13534)$',
     r'$m \;=\; 0.00034\times 1.06222 + 0.13534\times 0.23296 \;=\; 0.0319$',
     r'$v \;=\; 1 - 0.0169 \;=\; 0.9831$')
print('eqs ok')
