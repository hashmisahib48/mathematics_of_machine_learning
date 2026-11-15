# Equation and matrix images of Chapter 10. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq, matimg
import matplotlib.pyplot as plt
T=r'^{\mathrm{T}}'; X=r'\mathbf{X}'; y=r'\mathbf{y}'; w=r'\mathbf{w}'; wh=r'\hat{\mathbf{w}}'; Xc=r'\mathbf{X}_c'; yc=r'\mathbf{y}_c'; I=r'\mathbf{I}'
E={
# 10.3
'e1031': rf'$J(w_1,\ldots,w_p,b) \;=\; \sum_{{i=1}}^{{n}}\,\left(y_i - b - {w}{T}\mathbf{{x}}_i\right)^2 \;+\; \alpha\sum_{{j=1}}^{{p}} w_j^{{\,2}}$',
'e1032': rf'$b \;=\; \bar{{y}}-\bar{{\mathbf{{x}}}}{T}{w}$',
'e1033': rf'$J({w}) \;=\; \Vert {yc}-{Xc}{w}\Vert^2 \;+\; \alpha\,\Vert {w}\Vert^2$',
'e1035': rf'$\left({Xc}{T}{Xc} + \alpha{I}\right){wh} \;=\; {Xc}{T}{yc}$',
'e1036': rf'${wh} \;=\; \left({Xc}{T}{Xc} + \alpha{I}\right)^{{-1}}{Xc}{T}{yc}$',
'e1037': rf'$J({w}) \;=\; \left\Vert \binom{{{yc}}}{{\mathbf{{0}}}} - \binom{{{Xc}}}{{\sqrt{{\alpha}}\,{I}}}\,{w}\right\Vert^2$',
'e1038': rf'${wh} \;=\; {Xc}{T}\left({Xc}{Xc}{T} + \alpha{I}\right)^{{-1}}{yc}$',
'e1039': rf'$\Vert {yc}-{Xc}{w}\Vert^2 \;=\; \mathrm{{SSE}}_{{\min}} \;+\; ({w}-{wh}_{{0}}){T}\,{Xc}{T}{Xc}\,({w}-{wh}_{{0}})$',
# 10.4
'e1041': rf'${Xc} \;=\; \mathbf{{U}}\,\Sigma\,\mathbf{{V}}{T}\,, \qquad {Xc}{T}{Xc} + \alpha{I} \;=\; \mathbf{{V}}\left(\Sigma^2+\alpha{I}\right)\mathbf{{V}}{T}$',
'e1042': rf'$\left({Xc}{T}{Xc}+\alpha{I}\right)^{{-1}} \;=\; \mathbf{{V}}\,\left(\Sigma^2+\alpha{I}\right)^{{-1}}\mathbf{{V}}{T}$',
'e1043': rf'${wh} \;=\; \sum_{{j=1}}^{{r}} \dfrac{{\sigma_j}}{{\sigma_j^{{\,2}}+\alpha}}\,\left(\mathbf{{u}}_j{T}{yc}\right)\mathbf{{v}}_j$',
'e1044': rf'${wh} \;=\; \sum_{{j=1}}^{{r}} f_j\,\dfrac{{\mathbf{{u}}_j{T}{yc}}}{{\sigma_j}}\,\mathbf{{v}}_j\,, \qquad f_j \;=\; \dfrac{{\sigma_j^{{\,2}}}}{{\sigma_j^{{\,2}}+\alpha}}$',
'e1045': rf'$\hat{{\mathbf{{y}}}}_c \;=\; {Xc}{wh} \;=\; \sum_{{j=1}}^{{r}} f_j\,\left(\mathbf{{u}}_j{T}{yc}\right)\mathbf{{u}}_j$',
'e1046': rf'$\mathrm{{df}}(\alpha) \;=\; \sum_{{j=1}}^{{r}} f_j \;=\; \sum_{{j=1}}^{{r}} \dfrac{{\sigma_j^{{\,2}}}}{{\sigma_j^{{\,2}}+\alpha}}$',
'e1047': [r'$\mathrm{Var}(c_j^{\,\mathrm{OLS}}) \;=\; \dfrac{\sigma^2}{\lambda_j}\,, \qquad \mathrm{Var}(c_j^{\,\mathrm{ridge}}) \;=\; \dfrac{\sigma^2\,\lambda_j}{(\lambda_j+\alpha)^2}$'],
# 10.5
'e1051': r'$w \;=\; \dfrac{S_{xy}}{S_{xx}+\alpha}\,, \qquad b \;=\; \bar{y}-w\,\bar{x}$',
# 10.9
'e1091': rf'$e_i \;=\; y_i - \hat{{y}}_{{-i}}\,, \qquad \mathrm{{LOO}} \;=\; \dfrac{{1}}{{n}}\sum_{{i=1}}^{{n}} e_i^{{\,2}}$',
'e1092': rf'$\hat{{\mathbf{{y}}}} \;=\; \mathbf{{H}}_\alpha\,{y}\,, \qquad \mathbf{{H}}_\alpha \;=\; \dfrac{{1}}{{n}}\mathbf{{1}}\mathbf{{1}}{T} + {Xc}\left({Xc}{T}{Xc}+\alpha{I}\right)^{{-1}}{Xc}{T}$',
'e1093x': [r'$\hat{y}_{-i} \;=\; \sum_{k\neq i} H_{ik}\,y_k \;+\; H_{ii}\,\hat{y}_{-i}$', r'$\hat{y}_{-i} \;=\; \hat{y}_i - H_{ii}\,y_i + H_{ii}\,\hat{y}_{-i}$', r'$\hat{y}_{-i}\,(1-H_{ii}) \;=\; \hat{y}_i - H_{ii}\,y_i$'],
'e1094': r'$e_i \;=\; y_i-\hat{y}_{-i} \;=\; \dfrac{y_i-\hat{y}_i}{1-H_{ii}} \;=\; \dfrac{r_i}{1-h_{ii}}$',
'e1095': r'$h_{ii} \;=\; \dfrac{1}{n} + \sum_{j=1}^{r} f_j\,u_{ij}^{\,2}$',
# 10.11
'e10111': rf'$p({w}\mid{y}) \;\propto\; \exp\!\left(-\dfrac{{\Vert {yc}-{Xc}{w}\Vert^2}}{{2\sigma^2}}\right)\exp\!\left(-\dfrac{{\Vert {w}\Vert^2}}{{2\tau^2}}\right)$',
'e10112': rf'$-2\sigma^2\ln p({w}\mid{y}) \;=\; \Vert {yc}-{Xc}{w}\Vert^2 + \dfrac{{\sigma^2}}{{\tau^2}}\Vert {w}\Vert^2 + \mathrm{{const}}$',
'e10113': r'$\alpha \;=\; \dfrac{\sigma^2}{\tau^2}$',
}
single={k:v for k,v in E.items() if not k.endswith('x') and not (k=='e1093x')}
mkeq({k:v for k,v in single.items() if k not in ('e1093x',)},'.')
# three-line chain for the LOO derivation
fig=plt.figure(figsize=(.01,.01)); fig.text(0,0,'\n'.join(E['e1093x']),fontsize=11.5,linespacing=2.4); fig.savefig('e1093.png',dpi=300,bbox_inches='tight',pad_inches=0.02,facecolor='white'); plt.close(fig)
# gradient (two-term) equation
mkeq({'e1034': rf'$\nabla J({w}) \;=\; -2\,{Xc}{T}\left({yc}-{Xc}{w}\right) \;+\; 2\alpha\,{w}$'},'.')
M=lambda rows:[[str(v) for v in r] for r in rows]
matimg('m1052',[rf'${Xc}{T}{Xc}=$',M([[10,9],[9,10]]),rf'$,\quad {Xc}{T}{yc}=$',M([[17],[14]])])
matimg('m1053',[rf'${wh}\;(\alpha=0)=\frac{{1}}{{19}}$',M([[10,-9],[-9,10]]),M([[17],[14]]),r'$=\frac{1}{19}$',M([[44],[-13]]),'$=$',M([['2.3158'],['-0.6842']])])
matimg('m1054',[rf'${wh}\;(\alpha=1)=\frac{{1}}{{40}}$',M([[11,-9],[-9,11]]),M([[17],[14]]),r'$=\frac{1}{40}$',M([[61],[1]]),'$=$',M([['1.525'],['0.025']])])
print('eqs ok')
