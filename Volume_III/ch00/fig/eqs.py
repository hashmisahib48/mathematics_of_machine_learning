# Equation images of the refresher. Run: python3 fig/eqs.py
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
import matplotlib.pyplot as plt
T = r'^{\mathrm{T}}'
E = {
 'e0021': [r'$\mathbf{A} \;=\; \mathbf{U}\,\mathbf{\Sigma}\,\mathbf{V}' + T + r' \;=\; \sum_{j}\sigma_j\,\mathbf{u}_j\mathbf{v}_j' + T + r'\,, \qquad \mathbf{A}_k \;=\; \sum_{j=1}^{k}\sigma_j\,\mathbf{u}_j\mathbf{v}_j' + T + r'$',
           r'$\min_{\mathrm{rank}\,\mathbf{B}\leq k}\,\|\mathbf{A}-\mathbf{B}\|_F^2 \;=\; \|\mathbf{A}-\mathbf{A}_k\|_F^2 \;=\; \sigma_{k+1}^2+\sigma_{k+2}^2+\cdots$'],
 'e0022': r'$\lambda_{\min} \;\leq\; R(\mathbf{x}) \;=\; \dfrac{\mathbf{x}' + T + r'\mathbf{S}\mathbf{x}}{\mathbf{x}' + T + r'\mathbf{x}} \;\leq\; \lambda_{\max}\,, \qquad R(\mathbf{q}_j)=\lambda_j$',
 'e0023': r'$\mathbf{S}\mathbf{x} \;=\; \lambda\,\mathbf{D}\mathbf{x} \quad\Longleftrightarrow\quad \left(\mathbf{D}^{-1/2}\mathbf{S}\,\mathbf{D}^{-1/2}\right)\mathbf{g} \;=\; \lambda\,\mathbf{g}\,, \quad \mathbf{g}=\mathbf{D}^{1/2}\mathbf{x}$',
 'e0031': r'$\mathcal{L}(\mathbf{w},\boldsymbol{\lambda}) \;=\; f(\mathbf{w}) + \sum_{i=1}^{m}\lambda_i\,g_i(\mathbf{w})\,, \qquad \nabla f(\mathbf{w}) + \sum_{i}\lambda_i\nabla g_i(\mathbf{w}) \;=\; \mathbf{0}$',
 'e0041': r'$p(\mathbf{x}) \;=\; (2\pi)^{-d/2}(\det\Sigma)^{-1/2}\exp\!\left[-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})' + T + r'\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right]$',
 'e0042': [r'$\mathrm{KL}(q\,\|\,\gamma) \;=\; \sum_{k} q_k\,\ln\dfrac{q_k}{\gamma_k} \qquad \mathrm{or} \qquad \int q(x)\,\ln\dfrac{q(x)}{\gamma(x)}\,dx \;\geq\; 0$'],
 'e0043': r'$\mathrm{MI}(a,b) \;=\; \mathrm{KL}\!\left(P_{ab}\,\|\,P_a P_b\right) \;=\; \sum_{i,j}P_{ij}\,\ln\dfrac{P_{ij}}{P_{i\cdot}\,P_{\cdot j}}$',
 'e0044': r'$\mathrm{KL}\!\left(N(\mu_1,\sigma_1^2)\,\|\,N(\mu_2,\sigma_2^2)\right) \;=\; \ln\dfrac{\sigma_2}{\sigma_1} + \dfrac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2} - \dfrac{1}{2}$',
 'e0081': r'$\mathbf{f}' + T + r'\mathbf{L}\mathbf{f} \;=\; \sum_{i<j}W_{ij}\,(f_i-f_j)^2 \;\geq\; 0\,, \qquad \mathbf{L} = \mathbf{D}-\mathbf{W}$',
 'e0082': r'$\mathbf{L}_{\mathrm{sym}} \;=\; \mathbf{I}-\mathbf{D}^{-1/2}\mathbf{W}\mathbf{D}^{-1/2}\,, \qquad \mathbf{L}_{\mathrm{rw}} \;=\; \mathbf{I}-\mathbf{D}^{-1}\mathbf{W}$',
}
mkeq(E, '.')
print('eqs ok')
