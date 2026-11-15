# Equation images of Chapter 5. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq
import matplotlib.pyplot as plt
T=r'^{\mathrm{T}}'; X=r'\mathbf{X}'; y=r'\mathbf{y}'; w=r'\mathbf{w}'; wh=r'\hat{\mathbf{w}}'; Hm=r'\mathbf{H}'; g=r'\mathbf{g}'
E={
'e511': rf'$\min_{{{w}}}\; J({w})$',
'e512': rf'$\min_{{{w}}}\; J({w}) \quad \mathrm{{subject\ to}} \quad g({w}) \leq 0$',
'e521': rf'$t\,\mathbf{{a}} + (1-t)\,\mathbf{{b}} \;\in\; S \qquad \mathrm{{for\ all\ }}\mathbf{{a}},\mathbf{{b}}\in S,\ 0\leq t\leq 1$',
'e522': rf'$f(t\,\mathbf{{a}} + (1-t)\,\mathbf{{b}}) \;\leq\; t\,f(\mathbf{{a}}) + (1-t)\,f(\mathbf{{b}}) \qquad 0\leq t\leq 1$',
'e523': rf'$\mathbf{{u}}{T}{Hm}\,\mathbf{{u}} \;\geq\; 0 \qquad \mathrm{{for\ every\ vector\ }}\mathbf{{u}}$',
'e524': rf'$J({w},b) \;=\; \dfrac{{1}}{{2}}\Vert {w}\Vert^2 \;+\; C\sum_{{i=1}}^{{n}} \ln\!(1+e^{{-s_i({w}{T}\mathbf{{x}}_i + b)}})$',
'e531': rf'${w}_{{t+1}} \;=\; {w}_t \;-\; \eta\,\nabla J({w}_t)$',
'e532': r'$w_{t+1}-w^{*} \;=\; (1-\eta h)\,(w_t - w^{*}) \qquad \mathrm{for}\quad f(w)=\dfrac{h}{2}\,(w-w^{*})^2$',
'e533': rf'$\nabla J({w}) \;=\; -2\,{X}{T}({y}-{X}{w}), \qquad {Hm} \;=\; 2\,{X}{T}{X}$',
'e534': rf'${w}_t-{wh} \;=\; (\mathbf{{I}}-\eta{Hm})^{{t}}\,({w}_0-{wh})$',
'e535': r'$|1-\eta\lambda_k| < 1 \ \mathrm{for\ all\ }k \quad\Longleftrightarrow\quad 0 < \eta < \dfrac{2}{\lambda_{\max}}$',
'e536': r'$|1-\eta\lambda_{\min}| \;=\; 1-\dfrac{\lambda_{\min}}{\lambda_{\max}} \;=\; 1-\dfrac{1}{\kappa} \quad \mathrm{for}\quad \eta=\dfrac{1}{\lambda_{\max}}, \qquad \kappa=\dfrac{\lambda_{\max}}{\lambda_{\min}}$',
'e541': rf'$J({w}) \;=\; \dfrac{{1}}{{n}}\sum_{{i=1}}^{{n}} \ell_i({w}), \qquad \ell_i({w}) \;=\; \dfrac{{1}}{{2}}(y_i-{w}{T}\mathbf{{x}}_i)^2$',
'e542': rf'${g}_B \;=\; \dfrac{{1}}{{|B|}}\sum_{{i\in B}} \nabla\ell_i({w}), \qquad \mathbb{{E}}[{g}_B] \;=\; \nabla J({w})$',
'e543': rf'${w}_{{t+1}} \;=\; {w}_t + \eta_t\, r_i\,\mathbf{{x}}_i, \qquad r_i = y_i - {w}_t{T}\mathbf{{x}}_i$',
'e545': rf'$\mathbf{{v}}_{{t+1}} \;=\; \mu\,\mathbf{{v}}_t - \eta\,{g}_t, \qquad {w}_{{t+1}} \;=\; {w}_t + \mathbf{{v}}_{{t+1}}$',
'e551': rf'${w}_{{t+1}} \;=\; {w}_t \;-\; {Hm}^{{-1}}\,{g}_t, \qquad {g}_t=\nabla J({w}_t),\quad {Hm}=\nabla^2 J({w}_t)$',
'e552': r'$w_{t+1} \;=\; w_t \;-\; \dfrac{f^{\prime}(w_t)}{f^{\prime\prime}(w_t)}$',
'e553': rf'$\mathbf{{B}}_{{t+1}}\,({w}_{{t+1}}-{w}_t) \;=\; \nabla J({w}_{{t+1}}) - \nabla J({w}_t)$',
'e561': rf'$w_j \;\leftarrow\; \dfrac{{\mathbf{{x}}_j{T}\,\mathbf{{r}}_{{(-j)}}}}{{\mathbf{{x}}_j{T}\mathbf{{x}}_j}}, \qquad \mathbf{{r}}_{{(-j)}} \;=\; {y} - \sum_{{k\neq j}} w_k\,\mathbf{{x}}_k$',
'e562': rf'$J(w_j) \;=\; \Vert\, \mathbf{{r}}_{{(-j)}} - w_j\,\mathbf{{x}}_j \Vert^2$',
'e571': rf'$L({w},\lambda) \;=\; f({w}) + \lambda\, g({w})$',
'e572': rf'$\nabla f({w}) + \lambda\,\nabla g({w}) \;=\; \mathbf{{0}}, \qquad g({w}) = 0$',
'e573': r'$L(\mathbf{w},\lambda) \;=\; (w_1-3)^2+(w_2-4)^2 + \lambda\,(w_1^2+w_2^2-t^2)$',
'e574': r'$w_1 \;=\; \dfrac{3}{1+\lambda}, \qquad w_2 \;=\; \dfrac{4}{1+\lambda}$',
'e575': r'$\sqrt{w_1^2+w_2^2} \;=\; \dfrac{5}{1+\lambda} \;=\; t \quad\Longrightarrow\quad \lambda \;=\; \dfrac{5}{t}-1$',
'e576': rf'$f({w}) + \lambda\,\Vert{w}\Vert^2$',
'e577': r'$\min\; f(\mathbf{w}) \quad \mathrm{subject\ to} \quad |w_1|+|w_2| \leq t$',
'e578': r'$f(\mathbf{w}) + \mu\,(|w_1|+|w_2|)$',
'e579': r'$w_j \;=\; \mathrm{sign}(c_j)\,\max\!(|c_j| - \mu/2,\; 0)$',
'e5710': r'$q(\lambda) \;=\; \min_{\mathbf{w}}\,\left[f(\mathbf{w})+\lambda\,g(\mathbf{w})\right] \;\leq\; f(\mathbf{v}) \quad \mathrm{for\ every\ feasible\ }\mathbf{v}\ \mathrm{and}\ \lambda\geq 0$',
}
mkeq(E,'.')
def ml(name,lines,ls=2.5):
    fig=plt.figure(figsize=(.01,.01)); fig.text(0,0,'\n'.join(lines),fontsize=11.5,linespacing=ls)
    fig.savefig(name+'.png',dpi=300,bbox_inches='tight',pad_inches=0.02,facecolor='white'); plt.close(fig)
ml('e544',[r'$\eta_t = \eta_0 \qquad \mathrm{(constant)}$',r'$\eta_t = \dfrac{\eta_0}{t^{\,p}} \qquad \mathrm{(invscaling)}$',r'$\eta_t = \dfrac{1}{\alpha\,(t+t_0)} \qquad \mathrm{(optimal)}$'])
ml('e546',[rf'$\mathbf{{m}}_t=\beta_1\mathbf{{m}}_{{t-1}}+(1-\beta_1)\,{g}_t, \qquad \mathbf{{s}}_t=\beta_2\mathbf{{s}}_{{t-1}}+(1-\beta_2)\,{g}_t^{{\,2}}$',
  rf'${w}_{{t+1}} \;=\; {w}_t - \eta_t\,\dfrac{{\mathbf{{m}}_t}}{{\sqrt{{\mathbf{{s}}_t}}+\varepsilon}}, \qquad \eta_t=\eta\,\dfrac{{\sqrt{{1-\beta_2^{{\,t}}}}}}{{1-\beta_1^{{\,t}}}}$'],2.6)
print('eqs ok')
