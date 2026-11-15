# Equation images of Chapter 12. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq
T=r'^{\mathrm{T}}'; X=r'\mathbf{X}'; y=r'\mathbf{y}'; w=r'\mathbf{w}'; m=r'\mathbf{m}'; S=r'\mathbf{S}'; A=r'\mathbf{A}'; xs=r'\mathbf{x}_*'
E={
'e1231': rf'$y_i \;=\; {w}{T}\mathbf{{x}}_i + \varepsilon_i\,, \qquad \varepsilon_i \sim \mathcal{{N}}\!\left(0,\,\dfrac{{1}}{{\alpha}}\right)$',
'e1232': rf'$p({y}\,|\,{w}) \;=\; \left(\dfrac{{\alpha}}{{2\pi}}\right)^{{n/2}} \exp\!\left(-\dfrac{{\alpha}}{{2}}\,\Vert {y}-{X}{w}\Vert^2\right)$',
'e1233': rf'$p({w}) \;=\; \left(\dfrac{{\lambda}}{{2\pi}}\right)^{{p/2}} \exp\!\left(-\dfrac{{\lambda}}{{2}}\,\Vert {w}\Vert^2\right)$',
'e1234': rf'$p({w}\,|\,{y}) \;=\; \dfrac{{p({y}\,|\,{w})\;p({w})}}{{p({y})}} \;\propto\; p({y}\,|\,{w})\;p({w})$',
'e1235': rf'$\ln p({w}\,|\,{y}) \;=\; -\dfrac{{\alpha}}{{2}}\,\Vert {y}-{X}{w}\Vert^2 \;-\; \dfrac{{\lambda}}{{2}}\,\Vert {w}\Vert^2 \;+\; \mathrm{{const}}$',
'e1236': rf'$\hat{{{w}}}_{{\mathrm{{MAP}}}} \;=\; \arg\min_{{{w}}}\;\left[\,\Vert {y}-{X}{w}\Vert^2 \;+\; \dfrac{{\lambda}}{{\alpha}}\,\Vert {w}\Vert^2\,\right]$',
'e1237': rf'${A} \;=\; \lambda\,\mathbf{{I}} + \alpha\,{X}{T}{X}$',
'e1238': rf'$\ln p({w}\,|\,{y}) \;=\; -\dfrac{{1}}{{2}}\,{w}{T}{A}\,{w} \;+\; \alpha\,{w}{T}{X}{T}{y} \;+\; \mathrm{{const}}$',
'e1239': rf'$-\dfrac{{1}}{{2}}\,({w}-{m}){T}{A}\,({w}-{m}) \;=\; -\dfrac{{1}}{{2}}\,{w}{T}{A}\,{w} \;+\; {w}{T}{A}\,{m} \;-\; \dfrac{{1}}{{2}}\,{m}{T}{A}\,{m}$',
'e12310': rf'${A}\,{m} \;=\; \alpha\,{X}{T}{y}\,, \qquad {m} \;=\; \alpha\,{A}^{{-1}}{X}{T}{y}$',
'e12311': [rf'$p({w}\,|\,{y}) \;=\; \mathcal{{N}}({m},\,{S})\,, \qquad {S} \;=\; \left(\lambda\mathbf{{I}} + \alpha{X}{T}{X}\right)^{{-1}}$', rf'${m} \;=\; \alpha\,{S}{X}{T}{y} \;=\; \left({X}{T}{X} + \dfrac{{\lambda}}{{\alpha}}\mathbf{{I}}\right)^{{-1}}{X}{T}{y}$'],
'e12312': r'$m \;=\; \dfrac{\alpha\,S_{xy}}{\lambda+\alpha\,S_{xx}}\,, \qquad \mathrm{variance}\;=\; \dfrac{1}{\lambda+\alpha\,S_{xx}}$',
'e1241': rf'$\mathrm{{E}}\left[{w}{T}{xs}\right] \;=\; {m}{T}{xs}\,, \qquad \mathrm{{Var}}\left[{w}{T}{xs}\right] \;=\; {xs}{T}{S}\,{xs}$',
'e1242': rf'$y_* \;\sim\; \mathcal{{N}}\!\left({m}{T}{xs}\,,\;\; \dfrac{{1}}{{\alpha}} + {xs}{T}{S}\,{xs}\right)$',
'e1243': r'$\mathrm{Var}[y_*] \;=\; \dfrac{1}{\alpha} \;+\; \dfrac{x_*^{2}}{\lambda+\alpha\,S_{xx}}$',
'e1244': rf'$p({y}\,|\,\alpha,\lambda) \;=\; \mathcal{{N}}\!\left({y}\;|\;\mathbf{{0}}\,,\;\; \dfrac{{1}}{{\alpha}}\mathbf{{I}} + \dfrac{{1}}{{\lambda}}{X}{X}{T}\right)$',
'e1245': [rf'$\ln p({y}\,|\,\alpha,\lambda) \;=\; \dfrac{{1}}{{2}}\left[\,p\ln\lambda + n\ln\alpha - n\ln 2\pi - \ln|{A}|\,\right]$', rf'$\qquad\qquad -\,\dfrac{{\alpha}}{{2}}\Vert {y}-{X}{m}\Vert^2 - \dfrac{{\lambda}}{{2}}\Vert {m}\Vert^2$'],
'e1246': [rf'$\dfrac{{\partial \ln p}}{{\partial \lambda}} \;=\; \dfrac{{1}}{{2}}\left(\dfrac{{p}}{{\lambda}} - \Vert {m}\Vert^2 - \mathrm{{tr}}\,{S}\right) \;=\; 0$', rf'$\dfrac{{\partial \ln p}}{{\partial \alpha}} \;=\; \dfrac{{1}}{{2}}\left(\dfrac{{n}}{{\alpha}} - \Vert {y}-{X}{m}\Vert^2 - \mathrm{{tr}}\,({S}{X}{T}{X})\right) \;=\; 0$'],
'e1247': r'$\gamma \;=\; \sum_{j=1}^{p}\,\dfrac{\alpha\,d_j^{2}}{\lambda+\alpha\,d_j^{2}} \;=\; p-\lambda\,\mathrm{tr}\,\mathbf{S}\,, \qquad \mathrm{tr}(\mathbf{S}\mathbf{X}^{\mathrm{T}}\mathbf{X}) \;=\; \dfrac{\gamma}{\alpha}$',
'e1248': rf'$\lambda \;=\; \dfrac{{\gamma}}{{\Vert {m}\Vert^2}}\,, \qquad \alpha \;=\; \dfrac{{n-\gamma}}{{\Vert {y}-{X}{m}\Vert^2}}$',
'e1249': r'$\ln p(\mathbf{y}\,|\,\alpha,\lambda) \;+\; a_\lambda\ln\lambda - b_\lambda\,\lambda \;+\; a_\alpha\ln\alpha - b_\alpha\,\alpha$',
'e12410': rf'$\lambda \;=\; \dfrac{{\gamma + 2a_\lambda}}{{\Vert {m}\Vert^2 + 2b_\lambda}}\,, \qquad \alpha \;=\; \dfrac{{n-\gamma + 2a_\alpha}}{{\Vert {y}-{X}{m}\Vert^2 + 2b_\alpha}}$',
'e1291': r'$p(\mathbf{w}) \;=\; \prod_{j=1}^{p}\mathcal{N}\!\left(w_j\,|\,0,\,\dfrac{1}{\lambda_j}\right)$',
'e1292': rf'${S} \;=\; \left(\mathbf{{\Lambda}} + \alpha{X}{T}{X}\right)^{{-1}}, \qquad {m} \;=\; \alpha\,{S}{X}{T}{y}\,, \qquad \mathbf{{\Lambda}} = \mathrm{{diag}}(\lambda_1,\dots,\lambda_p)$',
'e1293': rf'$\gamma_j \;=\; 1-\lambda_j S_{{jj}}\,, \qquad \lambda_j \;=\; \dfrac{{\gamma_j + 2a_\lambda}}{{m_j^{{2}} + 2b_\lambda}}\,, \qquad \alpha \;=\; \dfrac{{n-\sum_{{j}}\,\gamma_j + 2a_\alpha}}{{\Vert {y}-{X}{m}\Vert^2 + 2b_\alpha}}$',
}
import matplotlib.pyplot as plt
for k in [k for k,v in E.items() if isinstance(v,list)]:
    l1,l2=E.pop(k)
    fig=plt.figure(figsize=(.01,.01)); fig.text(0,0,l1+'\n'+l2,fontsize=11.5,linespacing=(3.2 if k=='e1246' else 2.3)); fig.savefig(k+'.png',dpi=300,bbox_inches='tight',pad_inches=0.03,facecolor='white'); plt.close(fig)
mkeq(E,'.'); print('eqs ok')
