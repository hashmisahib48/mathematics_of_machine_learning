# Equation images of Chapter 46. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq
tr=r'\mathrm{tr}'
E={
# 46.3 empirical covariance and precision
'e4631': r'$\ell(\boldsymbol{\mu},\boldsymbol{\Theta})\;=\;\dfrac{1}{2}\left[\,\log\det\boldsymbol{\Theta}\;-\;\dfrac{1}{n}\sum_{i=1}^{n}(\mathbf{x}_i-\boldsymbol{\mu})^{T}\boldsymbol{\Theta}\,(\mathbf{x}_i-\boldsymbol{\mu})\;-\;p\log 2\pi\,\right]\,,\qquad \boldsymbol{\Theta}=\boldsymbol{\Sigma}^{-1}$',
'e4632': r'$\dfrac{1}{n}\sum_{i=1}^{n}(\mathbf{x}_i-\boldsymbol{\mu})^{T}\boldsymbol{\Theta}(\mathbf{x}_i-\boldsymbol{\mu})\;=\;\mathrm{tr}(\mathbf{S}\boldsymbol{\Theta})\;+\;(\bar{\mathbf{x}}-\boldsymbol{\mu})^{T}\boldsymbol{\Theta}(\bar{\mathbf{x}}-\boldsymbol{\mu})$',
'e4633': r'$\mathbf{S}\;=\;\dfrac{1}{n}\sum_{i=1}^{n}(\mathbf{x}_i-\bar{\mathbf{x}})(\mathbf{x}_i-\bar{\mathbf{x}})^{T}\,,\qquad \mathrm{E}\,\mathbf{S}\;=\;\dfrac{n-1}{n}\,\boldsymbol{\Sigma}$',
'e4634': r'$\dfrac{\partial}{\partial\boldsymbol{\Theta}}\left[\log\det\boldsymbol{\Theta}-\mathrm{tr}(\mathbf{S}\boldsymbol{\Theta})\right]\;=\;\boldsymbol{\Theta}^{-1}-\mathbf{S}\;=\;\mathbf{0}\quad\Rightarrow\quad\hat{\boldsymbol{\Sigma}}\;=\;\mathbf{S}$',
'e4635': r'$d^{2}(\mathbf{x})\;=\;(\mathbf{x}-\hat{\boldsymbol{\mu}})^{T}\hat{\boldsymbol{\Theta}}\,(\mathbf{x}-\hat{\boldsymbol{\mu}})\,,\qquad d^{2}\;\sim\;\chi^{2}_{p}\ \ \mathrm{for\ Gaussian\ data}$',
'e4636': r'$x_i\,|\,\mathrm{rest}\;\sim\;\mathcal{N}\!\left(-\sum_{j\neq i}\dfrac{\Theta_{ij}}{\Theta_{ii}}\,x_j\,,\ \ \dfrac{1}{\Theta_{ii}}\right)$',
'e4637': r'$\rho_{ij\cdot\mathrm{rest}}\;=\;-\dfrac{\Theta_{ij}}{\sqrt{\Theta_{ii}\,\Theta_{jj}}}$',
# 46.4 shrinkage
'e4641': r'$\hat{\boldsymbol{\Sigma}}_{\alpha}\;=\;(1-\alpha)\,\mathbf{S}\;+\;\alpha\,\hat{\mu}\,\mathbf{I}\,,\qquad \hat{\mu}=\dfrac{\mathrm{tr}\,\mathbf{S}}{p}\,,\qquad \lambda_j(\alpha)\;=\;(1-\alpha)\,\lambda_j+\alpha\,\hat{\mu}$',
'e4642': r'$R(\alpha)\;=\;\mathrm{E}\,\Vert\hat{\boldsymbol{\Sigma}}_{\alpha}-\boldsymbol{\Sigma}\Vert_F^{2}\;=\;(1-\alpha)^{2}\,\mathrm{E}\,\Vert\mathbf{S}-\boldsymbol{\Sigma}\Vert_F^{2}\;+\;\alpha^{2}\,\Vert m\mathbf{I}-\boldsymbol{\Sigma}\Vert_F^{2}\,,\qquad m=\dfrac{\mathrm{tr}\,\boldsymbol{\Sigma}}{p}$',
'e4643': r'$\alpha^{*}\;=\;\dfrac{\mathrm{E}\,\Vert\mathbf{S}-\boldsymbol{\Sigma}\Vert_F^{2}}{\mathrm{E}\,\Vert\mathbf{S}-m\mathbf{I}\Vert_F^{2}}$',
'e4644': r'$\hat{\beta}\;=\;\dfrac{1}{n^{2}p}\sum_{i=1}^{n}\Vert\mathbf{x}_i\mathbf{x}_i^{T}-\mathbf{S}\Vert_F^{2}\,,\qquad \hat{\delta}\;=\;\dfrac{\Vert\mathbf{S}-\hat{\mu}\mathbf{I}\Vert_F^{2}}{p}\,,\qquad \hat{\alpha}_{\mathrm{LW}}\;=\;\dfrac{\min(\hat{\beta},\hat{\delta})}{\hat{\delta}}$',
'e4645': r'$\hat{\rho}_{\mathrm{OAS}}\;=\;\min\!\left(\dfrac{\mathrm{tr}(\mathbf{S}^{2})+\mathrm{tr}^{2}(\mathbf{S})}{(n+1)\left[\mathrm{tr}(\mathbf{S}^{2})-\mathrm{tr}^{2}(\mathbf{S})/p\right]}\,,\ 1\right)$',
# 46.5 graphical lasso
'e4651': r'$J(\boldsymbol{\Theta})\;=\;-\log\det\boldsymbol{\Theta}\;+\;\mathrm{tr}(\mathbf{S}\boldsymbol{\Theta})\;+\;\alpha\sum_{i\neq j}|\Theta_{ij}|\,,\qquad \boldsymbol{\Theta}\ \mathrm{positive\ definite}$',
'e4652': r'$\boldsymbol{\Theta}^{-1}-\mathbf{S}\;=\;\alpha\,\boldsymbol{\Gamma}\,,\qquad \Gamma_{ij}=\mathrm{sign}\,\Theta_{ij}\ \ \mathrm{if}\ \ \Theta_{ij}\neq 0\,,\quad |\Gamma_{ij}|\leq 1\ \ \mathrm{otherwise}$',
'e4653': r'$\alpha_{\max}\;=\;\max_{i\neq j}|S_{ij}|$',
'e4654': [r'$\hat{\boldsymbol{\beta}}\;=\;\mathrm{arg\,min}_{\boldsymbol{\beta}}\ \ \dfrac{1}{2}\,\boldsymbol{\beta}^{T}\mathbf{W}_{11}\boldsymbol{\beta}\;-\;\boldsymbol{\beta}^{T}\mathbf{s}_{12}\;+\;\alpha\,\Vert\boldsymbol{\beta}\Vert_1$',
          r'$\mathbf{w}_{12}\;=\;\mathbf{W}_{11}\hat{\boldsymbol{\beta}}\,,\qquad \theta_{22}\;=\;\dfrac{1}{w_{22}-\mathbf{w}_{12}^{T}\hat{\boldsymbol{\beta}}}\,,\qquad \boldsymbol{\theta}_{12}\;=\;-\,\theta_{22}\,\hat{\boldsymbol{\beta}}$'],
'e4655': r'$\mathrm{gap}\;=\;\mathrm{tr}(\mathbf{S}\boldsymbol{\Theta})\;-\;p\;+\;\alpha\sum_{i\neq j}|\Theta_{ij}|\;<\;\mathtt{tol}$',
# 46.6 MCD
'e4661': r'$H^{*}\;=\;\mathrm{arg\,min}_{|H|=h}\ \det\mathbf{S}_H\,,\qquad h=\left\lceil\dfrac{n+p+1}{2}\right\rceil$',
'e4662': r'$d_i^{2}(H)\;=\;(\mathbf{x}_i-\bar{\mathbf{x}}_H)^{T}\,\mathbf{S}_H^{-1}\,(\mathbf{x}_i-\bar{\mathbf{x}}_H)\,,\qquad \sum_{i\in H}d_i^{2}(H)\;=\;h\,p$',
'e4663': r'$H_{2}\;=\;\{\,i:\ d_i^{2}(H_1)\ \mathrm{is\ among\ the}\ h\ \mathrm{smallest}\,\}\quad\Rightarrow\quad\det\mathbf{S}_{H_2}\;\leq\;\det\mathbf{S}_{H_1}$',
'e4664': [r'$\mathbf{S}_3=\dfrac{1}{h}\sum_{i\in H_2}(\mathbf{x}_i-\bar{\mathbf{x}}_{1})(\mathbf{x}_i-\bar{\mathbf{x}}_{1})^{T}$',
          r'$\mathrm{tr}(\mathbf{S}_1^{-1}\mathbf{S}_3)\;=\;\dfrac{1}{h}\sum_{i\in H_2}d_i^{2}(H_1)\;\leq\;\dfrac{1}{h}\sum_{i\in H_1}d_i^{2}(H_1)\;=\;p$',
          r'$\det\mathbf{S}_{H_2}\;\leq\;\det\mathbf{S}_3\;=\;\det\mathbf{S}_1\,\det(\mathbf{S}_1^{-1}\mathbf{S}_3)\;\leq\;\det\mathbf{S}_1\left[\dfrac{\mathrm{tr}(\mathbf{S}_1^{-1}\mathbf{S}_3)}{p}\right]^{p}\;\leq\;\det\mathbf{S}_1$'],
'e4665': r'$c(a)\;=\;\dfrac{a}{F_{\chi^{2}_{p+2}}\!\left(q_{a}\right)}\,,\qquad q_a=\chi^{2}_{p}\ \mathrm{quantile\ of\ order}\ a$',
'e4666': r'$\mathrm{keep}_i\;=\;\left[\,\dfrac{d_i^{2}}{c(h/n)}\;<\;\chi^{2}_{p,\,0.975}\,\right]\,,\qquad \hat{\boldsymbol{\Sigma}}\;=\;c(0.975)\,\mathbf{S}_{\mathrm{keep}}$',
}
mkeq(E,'.')
