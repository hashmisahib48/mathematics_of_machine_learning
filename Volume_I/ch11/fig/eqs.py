# Equation and matrix images of Chapter 11. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq, matimg
import matplotlib.pyplot as plt
T=r'^{\mathrm{T}}'; X=r'\mathbf{X}'; y=r'\mathbf{y}'; w=r'\mathbf{w}'; wh=r'\hat{\mathbf{w}}'; Xc=r'\mathbf{X}_c'; yc=r'\mathbf{y}_c'; r_=r'\mathbf{r}'; one=r'\mathbf{1}'
E={
'e1131': rf'$J(b,{w}) \;=\; \dfrac{{1}}{{2n}}\,\Vert {y}-b{one}-{X}{w}\Vert^2 \;+\; \alpha\,\Vert {w}\Vert_1\,, \qquad \Vert {w}\Vert_1=\sum_{{j=1}}^{{p}}|w_j|$',
'e1132': rf'$b \;=\; \bar{{y}}-\bar{{\mathbf{{x}}}}{T}{w}\,, \qquad J({w}) \;=\; \dfrac{{1}}{{2n}}\,\Vert {yc}-{Xc}{w}\Vert^2 \;+\; \alpha\,\Vert {w}\Vert_1$',
'e1133': r'$v_j \;=\; \dfrac{1}{n}\sum_{i=1}^{n} x_{ij}^{\,2}\,, \qquad c_j \;=\; \dfrac{1}{n}\sum_{i=1}^{n} x_{ij}\,y_i\qquad(\mathrm{centred\ data})$',
'e1134': r'$J(w) \;=\; \dfrac{1}{2}\,v\,w^{2} \;-\; c\,w \;+\; \alpha\,|w| \;+\; \mathrm{constant}$',
'e1135': [r'$v\,w - c + \alpha\,s \;=\; 0\,, \qquad s=\mathrm{sign}(w)\ \ \mathrm{if}\ w\neq 0\,,\qquad s\in[-1,\,1]\ \ \mathrm{if}\ w=0$'],
'e1136': r'$w \;=\; \dfrac{S(c,\alpha)}{v}\,, \qquad S(z,a) \;=\; \mathrm{sign}(z)\,\max\,(|z|-a,\;0)$',
'e1137': r'$\dfrac{1}{n}\,\mathbf{x}_j^{\mathrm{T}}\,(\mathbf{y}_c-\mathbf{X}_c\mathbf{w}) \;=\; \alpha\,s_j\,, \qquad s_j=\mathrm{sign}(w_j)\ \ \mathrm{or}\ \ s_j\in[-1,1]\ \mathrm{if}\ w_j=0$',
'e1138': r'$\alpha_{\max} \;=\; \max_{j}\ \dfrac{1}{n}\,\left|\,\mathbf{x}_j^{\mathrm{T}}\,\mathbf{y}_c\right|$',
'e1139': rf'$J({w}) \;=\; \mathrm{{constant}} + \dfrac{{1}}{{2}}\,({w}-{wh}){T}\,\mathbf{{G}}\,({w}-{wh}) + \alpha\,\Vert {w}\Vert_1\,, \qquad \mathbf{{G}}=\dfrac{{1}}{{n}}{Xc}{T}{Xc}$',
'e11310': rf'$z_j \;=\; \dfrac{{1}}{{n}}\,\mathbf{{x}}_j{T}{r_} + v_j\,w_j\,, \qquad w_j \;\leftarrow\; \dfrac{{S(z_j,\alpha)}}{{v_j}}\,, \qquad {r_}={yc}-{Xc}{w}$',
'e11311': r'$\mathbf{w}_A \;=\; \mathbf{G}_A^{-1}\,\left(\mathbf{c}_A-\alpha\,\mathbf{s}_A\right)$',
'e11312': rf'$\min_{{{w}}}\ \dfrac{{1}}{{2n}}\,\Vert {yc}-{Xc}{w}\Vert^2 \quad \mathrm{{subject\ to}}\quad \Vert {w}\Vert_1\leq t$',
'e1141': rf'$J(b,{w}) \;=\; \dfrac{{1}}{{2n}}\,\Vert {y}-b{one}-{X}{w}\Vert^2 \;+\; \alpha\rho\,\Vert {w}\Vert_1 \;+\; \dfrac{{1}}{{2}}\,\alpha\,(1-\rho)\,\Vert {w}\Vert_2^{{2}}$',
'e1142': r'$\lambda_1 \;=\; \alpha\rho\,, \quad \lambda_2 \;=\; \alpha(1-\rho)\,, \qquad \alpha=\lambda_1+\lambda_2\,, \quad \rho=\dfrac{\lambda_1}{\lambda_1+\lambda_2}$',
'e1143': r'$\mathbf{w}_A \;=\; \left(\mathbf{G}_A+\lambda_2\,\mathbf{I}\right)^{-1}\left(\mathbf{c}_A-\lambda_1\,\mathbf{s}_A\right)$',
'e1144': r'$w_j \;\leftarrow\; \dfrac{S(z_j,\,\lambda_1)}{v_j+\lambda_2}$',
'e1161': [rf'$\mathrm{{gap}}({w}) \;=\; \dfrac{{1}}{{2n}}\Vert{r_}\Vert^2+\alpha\Vert{w}\Vert_1-\dfrac{{1}}{{n}}\left(s\,{r_}{T}{yc}-\dfrac{{1}}{{2}}\,s^2\Vert{r_}\Vert^2\right)$',
          rf'$s \;=\; \min\left(1,\ \dfrac{{\alpha}}{{\max_j\,|\mathbf{{x}}_j{T}{r_}|/n}}\right)$'],
'e11101': rf'$c_j({w}) \;=\; \dfrac{{1}}{{n}}\,\mathbf{{x}}_j{T}\,({yc}-{Xc}{w})\,, \qquad |c_j| = \alpha\ \ \mathrm{{for\ every\ active}}\ j$',
'e11102': r'$\mathbf{w}_A(\alpha) \;=\; \mathbf{w}_A(\alpha_k)\;+\;(\alpha_k-\alpha)\;\mathbf{G}_A^{-1}\,\mathbf{s}_A$',
'e11103': [r'$\mathrm{IC}(\alpha) \;=\; n\ln\left(2\pi\hat\sigma^{2}\right) + \dfrac{\mathrm{RSS}(\alpha)}{\hat\sigma^{2}} + K\cdot \mathrm{df}(\alpha)$',
           r'$K=2\ (\mathrm{AIC})\,\ \ K=\ln n\ (\mathrm{BIC})\,, \qquad \hat\sigma^{2}=\dfrac{\mathrm{RSS}_{\mathrm{OLS}}}{n-p-1}\,, \qquad \mathrm{df}(\alpha)=\#\{j:\,w_j\neq 0\}$'],
'e11111': rf'$\min_{{{w}}}\ \Vert {y}-{X}{w}\Vert^2 \quad \mathrm{{subject\ to}}\quad \Vert {w}\Vert_0\leq k\,, \qquad \Vert {w}\Vert_0=\#\{{j:\,w_j\neq 0\}}$',
'e11112': rf'$j^{{*}} \;=\; \arg\max_{{j\notin S}}\ \left|\,\mathbf{{x}}_j{T}{r_}\right|$',
'e11113': rf'$\mathbf{{w}}_S \;=\; \arg\min\ \Vert {yc}-\mathbf{{X}}_{{S}}\,\mathbf{{w}}_S\Vert^2\,, \qquad {r_}={yc}-\mathbf{{X}}_{{S}}\,\mathbf{{w}}_S$',
'e11121': [r'$J(\mathbf{b},\mathbf{W}) \;=\; \dfrac{1}{2n}\,\Vert \mathbf{Y}-\mathbf{1}\mathbf{b}^{\mathrm{T}}-\mathbf{X}\mathbf{W}\Vert_{F}^{2} \;+\; \alpha\,\Vert \mathbf{W}\Vert_{2,1}$', r'$\Vert\mathbf{W}\Vert_{2,1}=\sum_{j=1}^{p}\sqrt{\sum_{k=1}^{K} W_{jk}^{2}}$'],
'e11122': r'$J \;=\; \dfrac{1}{2n}\,\Vert \mathbf{Y}-\mathbf{1}\mathbf{b}^{\mathrm{T}}-\mathbf{X}\mathbf{W}\Vert_{F}^{2} + \alpha\rho\,\Vert \mathbf{W}\Vert_{2,1} + \dfrac{1}{2}\,\alpha(1-\rho)\,\Vert \mathbf{W}\Vert_{F}^{2}$',
'e11123': r'$\mathbf{w}_j \;=\; \max\left(1-\dfrac{\alpha}{\Vert \mathbf{z}_j\Vert},\ 0\right)\dfrac{\mathbf{z}_j}{v_j}$',
'e11131': r'$p(w_j) \;=\; \dfrac{1}{2\beta}\,\exp\left(-\dfrac{|w_j|}{\beta}\right)$',
'e11132': rf'$-\ln p({w}\,|\,{y}) \;=\; \dfrac{{1}}{{2\sigma^2}}\,\Vert {yc}-{Xc}{w}\Vert^2 + \dfrac{{1}}{{\beta}}\Vert {w}\Vert_1 + \mathrm{{constant}}\,, \qquad \alpha=\dfrac{{\sigma^2}}{{n\beta}}$',
}
mkeq({k:v for k,v in E.items()},'.')
M=lambda rows:[[str(v) for v in r] for r in rows]
matimg('m1151',[rf'$\mathbf{{G}}=$',M([[2,1.6],[1.6,2]]),rf'$,\quad \mathbf{{c}}=$',M([[3.4],[3.0]]),rf'$,\quad \mathbf{{G}}^{{-1}}=\frac{{1}}{{1.44}}$',M([[2,-1.6],[-1.6,2]])])
matimg('m1152',[rf'$\mathbf{{w}}=\frac{{1}}{{1.44}}$',M([[2,-1.6],[-1.6,2]]),M([[r'3.4-1'],[r'3.0-1']]),r'$=\frac{1}{1.44}$',M([[r'2\times 2.4-1.6\times 2.0'],[r'-1.6\times 2.4+2\times 2.0']])])
matimg('m1153',[r'$\mathbf{w}=\frac{1}{1.44}$',M([['1.6'],['0.16']]),'$=$',M([[r'10/9'],[r'1/9']]),r'$=$',M([['1.1111'],['0.1111']])])
matimg('m1154',[rf'$\mathbf{{w}}=\frac{{1}}{{3.69}}$',M([[2.5,-1.6],[-1.6,2.5]]),M([[r'3.4-0.5'],[r'3.0-0.5']]),r'$=\frac{1}{3.69}$',M([['3.25'],['1.61']]),'$=$',M([['0.8808'],['0.4363']])])
print('eqs ok')
