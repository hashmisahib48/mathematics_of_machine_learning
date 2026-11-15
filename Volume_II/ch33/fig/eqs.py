# Equation images of Chapter 33. Run: python3 fig/eqs.py
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
T = r'^{\mathrm{T}}'
E = {
 # 33.3 many classes
 'e3331': r'$z^{(k)}_i \;=\; [\,y_i = k\,]\,, \qquad i = 1,\dots,n\,, \quad k = 1,\dots,K$',
 'e3332': r'$\hat{y}(\mathbf{x}) \;=\; \mathrm{arg\,max}_{\,k}\; f_k(\mathbf{x})$',
 'e3333': r'$\hat{P}(y = k \mid \mathbf{x}) \;=\; \dfrac{p_k(\mathbf{x})}{\sum_{j=1}^{K} p_j(\mathbf{x})}$',
 'e3334': r'$z^{(jk)}_i \;=\; [\,y_i = k\,] \quad \mathrm{for\ the\ samples\ with}\ y_i \in \{j, k\}\,, \qquad j < k$',
 'e3335': r'$v_k(\mathbf{x}) \;=\; \sum_{j<k} [\,f_{jk}(\mathbf{x}) > 0\,] \;+\; \sum_{l>k} [\,f_{kl}(\mathbf{x}) \leq 0\,]$',
 'e3336': r'$c_k(\mathbf{x}) \;=\; \sum_{j<k} f_{jk}(\mathbf{x}) \;-\; \sum_{l>k} f_{kl}(\mathbf{x})$',
 'e3337': r'$F_k(\mathbf{x}) \;=\; v_k(\mathbf{x}) + \dfrac{c_k(\mathbf{x})}{3\,(\,|c_k(\mathbf{x})| + 1\,)}\,, \qquad \hat{y}(\mathbf{x}) \;=\; \mathrm{arg\,max}_{\,k}\; F_k(\mathbf{x})$',
 'e3338': r'$\sum_{j<k}\,(n_j + n_k) \;=\; (K-1)\sum_{k=1}^{K} n_k \;=\; (K-1)\,n$',
 'e3339': r'$z^{(b)}_i \;=\; M_{y_i,\,b}\,, \qquad B \;=\; \lfloor\, K \cdot \mathtt{code\_size}\,\rfloor$',
 'e33310': r'$\hat{y}(\mathbf{x}) \;=\; \mathrm{arg\,min}_{\,k}\; \Vert\, \mathbf{s}(\mathbf{x}) - \mathbf{m}_k \,\Vert_2\,, \qquad \mathbf{s} = (s_1,\dots,s_B)$',
 # 33.4 many labels, many outputs
 'e3341': r'$P(y_1,\dots,y_L \mid \mathbf{x}) \;\approx\; \prod_{l=1}^{L} P(y_l \mid \mathbf{x})$',
 'e3342': r'$P(y_1,\dots,y_L \mid \mathbf{x}) \;=\; \prod_{l=1}^{L} P(y_l \mid \mathbf{x},\, y_1,\dots,y_{l-1})$',
 'e3343': r'$\mathrm{Hamming\ loss} \;=\; \dfrac{1}{nL}\sum_{i=1}^{n}\sum_{l=1}^{L} [\,\hat{y}_{il} \neq y_{il}\,]$',
 'e3344': r'$\mathrm{subset\ accuracy} \;=\; \dfrac{1}{n}\sum_{i=1}^{n} [\,\hat{\mathbf{y}}_i = \mathbf{y}_i\,]$',
 'e3345': r'$\hat{\mathbf{W}} \;=\; (\mathbf{X}' + T + r'\mathbf{X})^{-1}\mathbf{X}' + T + r'\mathbf{Y}\,, \qquad \hat{\mathbf{w}}_l \;=\; (\mathbf{X}' + T + r'\mathbf{X})^{-1}\mathbf{X}' + T + r'\mathbf{y}_l$',
 'e3346': r'$R^2 \;=\; \dfrac{1}{L}\sum_{l=1}^{L} R^2_l\,, \qquad R^2_l \;=\; 1 - \dfrac{\sum_i (y_{il}-\hat{y}_{il})^2}{\sum_i (y_{il}-\bar{y}_l)^2}$',
}
mkeq(E, '.')
print('eqs ok')
