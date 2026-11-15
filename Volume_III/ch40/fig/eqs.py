# Equation images of Chapter 40. Run: python3 fig/eqs.py
import sys, os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
N = r'\mathcal{N}'
E = {
 'e4031': r'$p(\mathbf{x}) \;=\; \sum_{k=1}^{K} \pi_k\,' + N + r'(\mathbf{x}\,|\,\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)\,, \qquad \pi_k \geq 0\,,\quad \sum_{k=1}^{K}\pi_k = 1$',
 'e4032': [r'$P(z=k) \;=\; \pi_k\,, \qquad p(\mathbf{x}\,|\,z=k) \;=\; '+N+r'(\mathbf{x}\,|\,\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)$', r'$p(\mathbf{x},\,z=k) \;=\; \pi_k\,'+N+r'(\mathbf{x}\,|\,\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)$'],
 'e4033': r'$\ell(\theta) \;=\; \sum_{i=1}^{n} \ln \sum_{k=1}^{K} \pi_k\,' + N + r'(\mathbf{x}_i\,|\,\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)$',
 'e4034': r'$\gamma_{ik} \;=\; P(z_i=k\,|\,\mathbf{x}_i) \;=\; \dfrac{\pi_k\,' + N + r'(\mathbf{x}_i\,|\,\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K}\pi_j\,' + N + r'(\mathbf{x}_i\,|\,\boldsymbol{\mu}_j,\boldsymbol{\Sigma}_j)}$',
 'e4041': r'$F(q,\theta) \;=\; \sum_{i=1}^{n}\sum_{k=1}^{K} q_{ik}\,\ln\dfrac{\pi_k\,' + N + r'(\mathbf{x}_i\,|\,\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)}{q_{ik}} \;\leq\; \ell(\theta)$',
 'e4042': r'$\ell(\theta) - F(q,\theta) \;=\; \sum_{i=1}^{n}\sum_{k=1}^{K} q_{ik}\,\ln\dfrac{q_{ik}}{\gamma_{ik}} \;=\; \sum_{i=1}^{n} \mathrm{KL}(q_i\,\|\,\gamma_i) \;\geq\; 0$',
 'e4043': r'$Q(\theta) \;=\; \sum_{i=1}^{n}\sum_{k=1}^{K} \gamma_{ik}\,\left[\,\ln\pi_k + \ln ' + N + r'(\mathbf{x}_i\,|\,\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)\right]$',
 'e4044': r'$\ln ' + N + r'(\mathbf{x}\,|\,\boldsymbol{\mu},\boldsymbol{\Sigma}) \;=\; -\dfrac{1}{2}\left[\,p\ln(2\pi) + \ln\det\boldsymbol{\Sigma} + (\mathbf{x}-\boldsymbol{\mu})^{T}\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\,\right]$',
 'e4045': r'$N_k \;=\; \sum_{i=1}^{n}\gamma_{ik}\,, \qquad \boldsymbol{\mu}_k \;=\; \dfrac{1}{N_k}\sum_{i=1}^{n}\gamma_{ik}\,\mathbf{x}_i$',
 'e4046': r'$\boldsymbol{\Sigma}_k \;=\; \dfrac{1}{N_k}\sum_{i=1}^{n}\gamma_{ik}\,(\mathbf{x}_i-\boldsymbol{\mu}_k)(\mathbf{x}_i-\boldsymbol{\mu}_k)^{T}$',
 'e4047': r'$\pi_k \;=\; \dfrac{N_k}{n}$',
 'e4048': r'$\ell(\theta_{\mathrm{new}}) \;\geq\; F(\gamma_{\mathrm{old}},\theta_{\mathrm{new}}) \;\geq\; F(\gamma_{\mathrm{old}},\theta_{\mathrm{old}}) \;=\; \ell(\theta_{\mathrm{old}})$',
 'e4051': r'$\gamma_{ik} \;=\; \dfrac{\exp\!\left(-\,\Vert\mathbf{x}_i-\boldsymbol{\mu}_k\Vert^{2}/2\sigma^{2}\right)}{\sum_{j=1}^{K}\exp\!\left(-\,\Vert\mathbf{x}_i-\boldsymbol{\mu}_j\Vert^{2}/2\sigma^{2}\right)}$',
 'e4052': r'$\gamma_{i1} \;=\; \dfrac{1}{1+\exp\!\left(-\,\dfrac{d_{2}^{2}-d_{1}^{2}}{2\sigma^{2}}\right)}\,, \qquad d_k = \Vert\mathbf{x}_i-\boldsymbol{\mu}_k\Vert$',
 'e4061': r'$q \;=\; (K-1) \;+\; Kp \;+\; c\,, \qquad c = \dfrac{Kp(p+1)}{2},\ \dfrac{p(p+1)}{2},\ Kp,\ K$',
 'e4062': r'$\mathrm{BIC} \;=\; -2\,\hat{\ell} + q\,\ln n\,, \qquad \mathrm{AIC} \;=\; -2\,\hat{\ell} + 2q$',
 'e4071': r'$\pi_k \;=\; v_k\prod_{j<k}(1-v_j)\,, \qquad v_k \sim \mathrm{Beta}(1,\,\gamma_0)$',
 'e4072': r'$v_k \,|\, \mathbf{X} \;\sim\; \mathrm{Beta}\!\left(1+N_k,\; \gamma_0+\sum_{j>k}N_j\right)$',
 'e4073': r'$\boldsymbol{\pi} \sim \mathrm{Dirichlet}(\gamma_0,\dots,\gamma_0) \quad\Longrightarrow\quad \pi_k \;=\; \dfrac{\gamma_0+N_k}{K\gamma_0+n}$',
 'e4081': r'$' + N + r'(x\,|\,\mu,\sigma^{2}) \;=\; \dfrac{1}{\sqrt{2\pi\sigma^{2}}}\exp\!\left(-\dfrac{(x-\mu)^{2}}{2\sigma^{2}}\right)$',
}
mkeq(E, '.')
print('eqs ok', len(E))
