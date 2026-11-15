# Equation images of Chapter 35. Run: python3 fig/eqs.py
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
T = r'^{\mathrm{T}}'
E = {
 # 35.3 one target
 'e3531': r'$\mathbf{w}_k \;=\; \mathrm{arg\,max}_{\,\Vert\mathbf{w}\Vert=1}\; (\mathbf{X}\mathbf{w})' + T + r'\mathbf{y} \;=\; \dfrac{\mathbf{X}' + T + r'\mathbf{y}}{\Vert\mathbf{X}' + T + r'\mathbf{y}\Vert}$',
 'e3532': r'$\dfrac{(\mathbf{X}\mathbf{w})' + T + r'\mathbf{y}}{n-1} \;=\; \mathrm{sd}(\mathbf{X}\mathbf{w})\;\mathrm{sd}(\mathbf{y})\;\mathrm{corr}(\mathbf{X}\mathbf{w},\,\mathbf{y})$',
 'e3533': r'$\mathbf{t}_k \;=\; \mathbf{X}\mathbf{w}_k$',
 'e3534': r'$\mathbf{p}_k \;=\; \dfrac{\mathbf{X}' + T + r'\mathbf{t}_k}{\mathbf{t}_k' + T + r'\mathbf{t}_k}$',
 'e3535': r'$q_k \;=\; \dfrac{\mathbf{y}' + T + r'\mathbf{t}_k}{\mathbf{t}_k' + T + r'\mathbf{t}_k}$',
 'e3536': r'$\mathbf{X} \;\leftarrow\; \mathbf{X} - \mathbf{t}_k\mathbf{p}_k' + T + r'\,, \qquad \mathbf{y} \;\leftarrow\; \mathbf{y} - q_k\,\mathbf{t}_k$',
 'e3537': r'$\mathbf{X} \;=\; \mathbf{T}\mathbf{P}' + T + r' + \mathbf{E}\,, \qquad \mathbf{y} \;=\; \mathbf{T}\mathbf{q} + \mathbf{f}$',
 'e3538': r'$\mathbf{T} \;=\; \mathbf{X}\mathbf{R}\,, \qquad \mathbf{R} \;=\; \mathbf{W}\,(\mathbf{P}' + T + r'\mathbf{W})^{-1}$',
 'e3539': r'$\hat{\mathbf{y}} \;=\; \mathbf{T}\mathbf{q} \;=\; \mathbf{X}\mathbf{B}\,, \qquad \mathbf{B} \;=\; \mathbf{R}\,\mathbf{q}$',
 'e35310': r'$\mathtt{coef\_}[j] \;=\; \dfrac{s_y}{s_j}\,B_j\,, \qquad \mathtt{intercept\_} \;=\; \bar{y}$',
 # 35.4 several targets
 'e3541': r'$(\mathbf{X}\mathbf{w})' + T + r'(\mathbf{Y}\mathbf{c}) \;=\; \mathbf{w}' + T + r'(\mathbf{X}' + T + r'\mathbf{Y})\,\mathbf{c} \;\leq\; \sigma_1\,, \qquad \Vert\mathbf{w}\Vert = \Vert\mathbf{c}\Vert = 1$',
 'e3542': r'$\mathbf{X}' + T + r'\mathbf{Y} \;=\; \mathbf{U}\,\mathbf{\Sigma}\,\mathbf{V}' + T + r'\,, \qquad \mathbf{w}_1 = \mathbf{u}_1\,, \quad \mathbf{c}_1 = \mathbf{v}_1$',
 'e3543': r'$\mathbf{w} \;\propto\; \mathbf{X}' + T + r'\mathbf{Y}\,\mathbf{c} \;\propto\; \mathbf{X}' + T + r'\mathbf{Y}\mathbf{Y}' + T + r'\mathbf{X}\,\mathbf{w}_{\mathrm{old}}$',
 'e3544': r'$\mathbf{w} \;=\; \dfrac{\mathbf{X}' + T + r'\mathbf{u}}{\mathbf{u}' + T + r'\mathbf{u}}\,, \qquad \mathbf{w} \;\leftarrow\; \dfrac{\mathbf{w}}{\Vert\mathbf{w}\Vert}\,, \qquad \mathbf{t} \;=\; \mathbf{X}\mathbf{w}$',
 'e3545': r'$\mathbf{c} \;=\; \dfrac{\mathbf{Y}' + T + r'\mathbf{t}}{\mathbf{t}' + T + r'\mathbf{t}}$',
 'e3546': r'$\mathbf{u} \;=\; \dfrac{\mathbf{Y}\mathbf{c}}{\mathbf{c}' + T + r'\mathbf{c}}$',
 'e3547': r'$\mathbf{X} \;\leftarrow\; \mathbf{X} - \mathbf{t}\mathbf{p}' + T + r'\,, \qquad \mathbf{Y} \;\leftarrow\; \mathbf{Y} - \mathbf{t}\mathbf{q}' + T + r'\,, \qquad \mathbf{q} \;=\; \dfrac{\mathbf{Y}' + T + r'\mathbf{t}}{\mathbf{t}' + T + r'\mathbf{t}}$',
 'e3548': r'$\mathbf{X} \;\leftarrow\; \mathbf{X} - \mathbf{t}\mathbf{p}' + T + r'\,, \qquad \mathbf{Y} \;\leftarrow\; \mathbf{Y} - \mathbf{u}\mathbf{d}' + T + r'\,, \qquad \mathbf{d} \;=\; \dfrac{\mathbf{Y}' + T + r'\mathbf{u}}{\mathbf{u}' + T + r'\mathbf{u}}$',
 'e3549': r'$\mathbf{w} \;=\; \mathbf{X}^{+}\mathbf{u}\,, \qquad \mathbf{c} \;=\; \mathbf{Y}^{+}\mathbf{t}\,, \qquad \mathbf{w}\,,\,\mathbf{c} \;\leftarrow\; \mathrm{unit\ length}$',
 'e35410': r'$\rho_1^{2} \;=\; \lambda_{\max}\!\left[\,(\mathbf{X}' + T + r'\mathbf{X})^{-1}\mathbf{X}' + T + r'\mathbf{Y}\,(\mathbf{Y}' + T + r'\mathbf{Y})^{-1}\mathbf{Y}' + T + r'\mathbf{X}\,\right]$',
}
mkeq(E, '.')
print('eqs ok')
