# Equation images of Chapter 34. Run: python3 fig/eqs.py
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
T = r'^{\mathrm{T}}'
E = {
 # 34.3 graph methods
 'e3431': r'$w_{ij} \;=\; \exp\!\left(-\gamma\,\Vert\,\mathbf{x}_i - \mathbf{x}_j\,\Vert^2\right)$',
 'e3432': r'$w_{ij} \;=\; [\,\mathbf{x}_j\ \mathrm{is\ among\ the}\ k\ \mathrm{nearest\ rows\ to}\ \mathbf{x}_i\,]\,, \qquad k = \mathtt{n\_neighbors}$',
 'e3433': r'$d_i \;=\; \sum_{j=1}^{n} w_{ij}\,, \qquad \mathbf{D} = \mathrm{diag}(d_1,\dots,d_n)\,, \qquad \mathbf{T} \;=\; \mathbf{D}^{-1}\mathbf{W}\,, \quad T_{ij} = \dfrac{w_{ij}}{d_i}$',
 'e3434': [r'$E(\mathbf{f}) \;=\; \dfrac{1}{2}\sum_{i,j}\, w_{ij}\,(f_i - f_j)^2 \;=\; \dfrac{1}{2}\sum_{i,j}\, w_{ij}\,(f_i^2 - 2f_if_j + f_j^2)$',
           r'$\qquad\qquad =\; \sum_i\, d_i f_i^2 - \sum_{i,j}\, w_{ij} f_i f_j \;=\; \mathbf{f}' + T + r'(\mathbf{D}-\mathbf{W})\,\mathbf{f}$'],
 'e3435': [r'$\dfrac{\partial E}{\partial \mathbf{f}_U} \;=\; 2\left[(\mathbf{D}-\mathbf{W})_{UU}\,\mathbf{f}_U + (\mathbf{D}-\mathbf{W})_{UL}\,\mathbf{f}_L\right] \;=\; \mathbf{0}$',
           r'$(\mathbf{D}-\mathbf{W})_{UU} = \mathbf{D}_{UU} - \mathbf{W}_{UU}\,, \qquad (\mathbf{D}-\mathbf{W})_{UL} = -\,\mathbf{W}_{UL}$'],
 'e3436': r'$\mathbf{f}_U \;=\; (\mathbf{D}_{UU} - \mathbf{W}_{UU})^{-1}\,\mathbf{W}_{UL}\,\mathbf{f}_L$',
 'e3437': [r'$\mathbf{F}_U^{(t+1)} \;=\; \mathbf{T}_{UU}\,\mathbf{F}_U^{(t)} + \mathbf{T}_{UL}\,\mathbf{Y}_L$',
           r'$\mathbf{F}_U \;=\; (\mathbf{I} - \mathbf{T}_{UU})^{-1}\,\mathbf{T}_{UL}\,\mathbf{Y}_L$'],
 'e3438': r'$Q(\mathbf{F}) \;=\; \dfrac{1}{2}\sum_{i,j}\, w_{ij}\,\Vert\, \mathbf{F}_i/\sqrt{d_i} - \mathbf{F}_j/\sqrt{d_j}\,\Vert^2 \;+\; \mu \sum_{i=1}^{n}\, \Vert\,\mathbf{F}_i - \mathbf{Y}_i\,\Vert^2$',
 'e3439': [r'$\mathbf{S} \;=\; \mathbf{D}^{-1/2}\,\mathbf{W}\,\mathbf{D}^{-1/2}$',
           r'$Q(\mathbf{F}) \;=\; \mathrm{tr}\left[\mathbf{F}' + T + r'(\mathbf{I}-\mathbf{S})\,\mathbf{F}\right] + \mu\,\Vert\,\mathbf{F}-\mathbf{Y}\,\Vert_F^2$'],
 'e34310': [r'$(\mathbf{I}-\mathbf{S})\,\mathbf{F} + \mu\,(\mathbf{F}-\mathbf{Y}) \;=\; \mathbf{0}$',
            r'$\mathbf{F}^{*} \;=\; (1-\alpha)\,(\mathbf{I} - \alpha\,\mathbf{S})^{-1}\,\mathbf{Y}\,, \qquad \alpha = \dfrac{1}{1+\mu}$'],
 'e34311': r'$\mathbf{F}^{(t+1)} \;=\; \alpha\,\mathbf{S}\,\mathbf{F}^{(t)} + (1-\alpha)\,\mathbf{Y}\,, \qquad \mathbf{F}^{(0)} = \mathbf{Y}$',
 'e34312': [r'$p_k(\mathbf{x}) \;=\; \sum_{i=1}^{n}\, w(\mathbf{x},\mathbf{x}_i)\,F_{ik}$',
            r'$\hat{P}(y = k \mid \mathbf{x}) \;=\; \dfrac{p_k(\mathbf{x})}{p_1(\mathbf{x}) + \dots + p_K(\mathbf{x})}$'],
 # 34.4 self-training
 'e3441': [r'$\hat{y}_i \;=\; \mathrm{arg\,max}_{\,k}\; \hat{p}_k(\mathbf{x}_i)\,, \qquad m_i \;=\; \max_k\, \hat{p}_k(\mathbf{x}_i)\,, \qquad i \in U_t$',
           r'$A_t \;=\; \{\, i \in U_t : m_i > \tau \,\} \quad \mathrm{or\ the}\ k_{\mathrm{best}}\ \mathrm{indices\ with\ the\ largest}\ m_i$'],
}
mkeq(E, '.')
print('eqs ok')
