# Equation and matrix images of Chapter 17. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq, matimg
import matplotlib.pyplot as plt
T=r'^{\mathrm{T}}'; X=r'\mathbf{X}'; w=r'\mathbf{w}'; xi=r'\mathbf{x}_i'; wh=r'\hat{\mathbf{w}}'
E={
# ---- 17.3 the perceptron
'e1731': rf'$f(\mathbf{{x}}) \;=\; {w}{T}\mathbf{{x}} + b \;=\; w_1x_1+\cdots+w_px_p + b\,, \qquad \hat{{y}} \;=\; \mathrm{{sign}}\,f(\mathbf{{x}})$',
'e1732': rf'$z_i \;=\; y_i\,f({xi}) \;=\; y_i\,({w}{T}{xi} + b)$',
'e1733': rf'$\mathrm{{if}}\;\; z_i \leq 0:\qquad {w} \;\leftarrow\; {w} + \eta\,y_i\,{xi}\,, \qquad b \;\leftarrow\; b + \eta\,y_i$',
'e1734': [rf'$z_i^{{\mathrm{{new}}}} \;=\; y_i\,\left[({w}+\eta y_i{xi}){T}{xi} + b + \eta y_i\right]$',
          rf'$\;=\; z_i + \eta\,y_i^2\,({xi}{T}{xi} + 1) \;=\; z_i + \eta\,(\Vert{xi}\Vert^2 + 1)$'],
# ---- 17.4 losses of the margin
'e1741': r'$L_{\mathrm{log}}(z) \;=\; \ln\left(1+e^{-z}\right) \;=\; -\ln\sigma(z)\,, \qquad \sigma(z) \;=\; \dfrac{1}{1+e^{-z}}$',
'e1742': rf'$\dfrac{{\partial L}}{{\partial {w}}} \;=\; L^{{\prime}}(z_i)\,y_i\,{xi}\,, \qquad \dfrac{{\partial L}}{{\partial b}} \;=\; L^{{\prime}}(z_i)\,y_i$',
# ---- 17.5 SGD
'e1751': rf'$J({w},b) \;=\; \dfrac{{1}}{{n}}\sum_{{i=1}}^{{n}} L\left(y_i,\,{w}{T}{xi}+b\right) \;+\; \alpha\,R({w})$',
'e1752': rf'$R({w}) \;=\; \dfrac{{1-\rho}}{{2}}\,\Vert{w}\Vert_2^2 \;+\; \rho\,\Vert{w}\Vert_1$',
'e1753': [rf'${w} \;\leftarrow\; \left(1-\eta\,(1-\rho)\,\alpha\right){w} \;-\; \eta\,L^{{\prime}}(z_i)\,y_i\,{xi}$',
          r'$b \;\leftarrow\; b \;-\; \eta\,L^{\prime}(z_i)\,y_i$'],
'e1754': r'$\eta_t \;=\; \eta_0$',
'e1755': r'$\eta_t \;=\; \dfrac{\eta_0}{t^{\,p}}$',
'e1756': r'$\eta_1 \;=\; \alpha^{-1/4}$',
'e1757': r'$\eta_t \;=\; \dfrac{1}{\alpha\,(t+t_0)} \;=\; \dfrac{1}{1/\eta_1 + \alpha\,(t-1)}\,, \qquad t_0 \;=\; \dfrac{1}{\alpha\,\eta_1} - 1 \;=\; \alpha^{-3/4}-1$',
'e1758': rf'$\bar{{{w}}}_t \;=\; \dfrac{{1}}{{t-a+1}}\sum_{{s=a}}^{{t}} {w}_s\,, \qquad t \geq a$',
# ---- 17.6 ridge classifier
'e1761': rf'$J({w},b) \;=\; \sum_{{i=1}}^{{n}} \left(t_i - {w}{T}{xi} - b\right)^2 + \alpha\,\Vert{w}\Vert^2\,, \qquad t_i \in \{{-1,\,+1\}}$',
'e1762': rf'${wh} \;=\; \left({X}_c{T}{X}_c + \alpha\,\mathbf{{I}}\right)^{{-1}}{X}_c{T}\mathbf{{t}}_c\,, \qquad b \;=\; \bar{{t}} - \bar{{\mathbf{{x}}}}{T}{wh}$',
'e1763': r'$(t_i - f_i)^2 \;=\; t_i^2\,(1 - t_i f_i)^2 \;=\; (1 - z_i)^2$',
'e1764': rf'$\hat{{y}} \;=\; \mathrm{{argmax}}_{{k}}\;\left({w}_k{T}\mathbf{{x}} + b_k\right)$',
# ---- 17.12 convergence proof
'e17121': [rf'$\mathbf{{u}}_{{k+1}}\!\cdot\mathbf{{u}}^{{*}} \;=\; \mathbf{{u}}_k\!\cdot\mathbf{{u}}^{{*}} + y_i\,\tilde{{\mathbf{{x}}}}_i\!\cdot\mathbf{{u}}^{{*}} \;\geq\; \mathbf{{u}}_k\!\cdot\mathbf{{u}}^{{*}} + \gamma$',
           r'$\Rightarrow\quad \mathbf{u}_k\!\cdot\mathbf{u}^{*} \;\geq\; k\,\gamma$'],
'e17122': [r'$\Vert\mathbf{u}_{k+1}\Vert^2 \;=\; \Vert\mathbf{u}_k\Vert^2 + 2\,y_i\,\mathbf{u}_k\!\cdot\tilde{\mathbf{x}}_i + \Vert\tilde{\mathbf{x}}_i\Vert^2 \;\leq\; \Vert\mathbf{u}_k\Vert^2 + R^2$',
           r'$\Rightarrow\quad \Vert\mathbf{u}_k\Vert^2 \;\leq\; k\,R^2$'],
'e17123': r'$k\,\gamma \;\leq\; \mathbf{u}_k\!\cdot\mathbf{u}^{*} \;\leq\; \Vert\mathbf{u}_k\Vert \;\leq\; \sqrt{k}\,R \qquad\Rightarrow\qquad k \;\leq\; \dfrac{R^2}{\gamma^2}$',
}
mkeq(E,'.')
# matrices for the by-hand examples
M=lambda rows:[[str(v) for v in r] for r in rows]
print('eqs ok')
