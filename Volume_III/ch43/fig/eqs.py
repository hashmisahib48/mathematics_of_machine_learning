# Equation and matrix images of Chapter 43. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq, matimg
E={
# 43.2  the common frame
'e4321': r'$\mathbf{X} \;\approx\; \mathbf{W}\,\mathbf{H}\,, \qquad \mathbf{x}_i \;\approx\; \sum_{t=1}^{k} w_{it}\,\mathbf{h}_t\,, \qquad \mathbf{W}\in\mathbb{R}^{n\times k}\,,\ \ \mathbf{H}\in\mathbb{R}^{k\times p}$',
# 43.3  NMF
'e4331': r'$\min_{\mathbf{W}\geq 0,\ \mathbf{H}\geq 0}\ \ J(\mathbf{W},\mathbf{H}) \;=\; \dfrac{1}{2}\,\Vert\mathbf{X}-\mathbf{W}\mathbf{H}\Vert_F^{2} \;=\; \dfrac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{p}\left(x_{ij}-\sum_{t=1}^{k}w_{it}h_{tj}\right)^{2}$',
'e4332': [r'$D_{\beta}(\mathbf{X}\,|\,\mathbf{W}\mathbf{H}) \;=\; \sum_{i,j} d_{\beta}(x_{ij}\,|\,y_{ij})\,, \qquad y_{ij}=(\mathbf{W}\mathbf{H})_{ij}$',
          r'$d_{\beta}(x\,|\,y) \;=\; \dfrac{x^{\beta}+(\beta-1)\,y^{\beta}-\beta\,x\,y^{\beta-1}}{\beta\,(\beta-1)}$'],
'e4333': r'$d_{2}(x\,|\,y)=\dfrac{1}{2}(x-y)^{2}\,, \qquad d_{1}(x\,|\,y)=x\log\dfrac{x}{y}-x+y\,, \qquad d_{0}(x\,|\,y)=\dfrac{x}{y}-\log\dfrac{x}{y}-1$',
'e4334': [r'$J \;=\; D_{\beta}(\mathbf{X}\,|\,\mathbf{W}\mathbf{H}) \;+\; \alpha_W\rho\,p\,\Vert\mathbf{W}\Vert_{1} \;+\; \alpha_H\rho\,n\,\Vert\mathbf{H}\Vert_{1}$',
          r'$\qquad\qquad +\;\dfrac{1}{2}\,\alpha_W(1-\rho)\,p\,\Vert\mathbf{W}\Vert_F^{2} \;+\; \dfrac{1}{2}\,\alpha_H(1-\rho)\,n\,\Vert\mathbf{H}\Vert_F^{2}\,, \qquad \rho=\mathtt{l1\_ratio}$'],
'e4335': r'$\nabla_{\mathbf{H}}J \;=\; \mathbf{W}^{T}\mathbf{W}\mathbf{H}-\mathbf{W}^{T}\mathbf{X} \;=\; \nabla^{+}-\nabla^{-}\,, \qquad \nabla_{\mathbf{W}}J \;=\; \mathbf{W}\mathbf{H}\mathbf{H}^{T}-\mathbf{X}\mathbf{H}^{T}$',
'e4336': r'$\mathbf{W} \;\leftarrow\; \mathbf{W}\odot\dfrac{\mathbf{X}\mathbf{H}^{T}}{\mathbf{W}\mathbf{H}\mathbf{H}^{T}}\,, \qquad \mathbf{H} \;\leftarrow\; \mathbf{H}\odot\dfrac{\mathbf{W}^{T}\mathbf{X}}{\mathbf{W}^{T}\mathbf{W}\mathbf{H}}$',
'e4337': [r'$\eta_{tj}=H_{tj}/(\mathbf{W}^{T}\mathbf{W}\mathbf{H})_{tj}\,, \qquad H_{tj}-\eta_{tj}\,(\nabla_{\mathbf{H}}J)_{tj} \;=\; H_{tj}-\eta_{tj}\left[(\mathbf{W}^{T}\mathbf{W}\mathbf{H})_{tj}-(\mathbf{W}^{T}\mathbf{X})_{tj}\right]$',
          r'$\qquad\qquad\qquad\qquad\qquad\qquad\ \ =\; H_{tj}-H_{tj}+\eta_{tj}\,(\mathbf{W}^{T}\mathbf{X})_{tj} \;=\; H_{tj}\,\dfrac{(\mathbf{W}^{T}\mathbf{X})_{tj}}{(\mathbf{W}^{T}\mathbf{W}\mathbf{H})_{tj}}$'],
'e4338': r'$H_{tj} \;\leftarrow\; H_{tj}\,\dfrac{\sum_{i} W_{it}\,x_{ij}/(\mathbf{W}\mathbf{H})_{ij}}{\sum_{i} W_{it}}\,, \qquad W_{it} \;\leftarrow\; W_{it}\,\dfrac{\sum_{j} H_{tj}\,x_{ij}/(\mathbf{W}\mathbf{H})_{ij}}{\sum_{j} H_{tj}}$',
'e4339': r'$H_{tj} \;\leftarrow\; H_{tj}\left[\dfrac{\left(\mathbf{W}^{T}\,[\,(\mathbf{W}\mathbf{H})^{\beta-2}\odot\mathbf{X}\,]\right)_{tj}}{\left(\mathbf{W}^{T}\,(\mathbf{W}\mathbf{H})^{\beta-1}\right)_{tj}}\right]^{\gamma(\beta)}$',
'e43310': r'$W_{it} \;\leftarrow\; \max\left\{0,\ W_{it}-\dfrac{(\mathbf{W}\mathbf{H}\mathbf{H}^{T}-\mathbf{X}\mathbf{H}^{T})_{it}}{(\mathbf{H}\mathbf{H}^{T})_{tt}}\right\}\,, \qquad t=1,\dots,k\ \ \mathrm{in\ turn}$',
# 43.4  FastICA
'e4341': r'$\mathbf{x}_i \;=\; \mathbf{A}\,\mathbf{s}_i\,, \qquad \mathbf{X} \;=\; \mathbf{S}\,\mathbf{A}^{T}\,, \qquad p(\mathbf{s}) \;=\; p_1(s_1)\,p_2(s_2)\cdots p_k(s_k)$',
'e4342': r'$\mathbf{z} \;=\; \mathbf{K}\,(\mathbf{x}-\boldsymbol{\mu})\,, \quad \mathbf{K}=\boldsymbol{\Lambda}^{-1/2}\mathbf{E}^{T}\,, \qquad \mathbf{I} \;=\; \mathrm{Cov}(\mathbf{z}) \;=\; (\mathbf{K}\mathbf{A})(\mathbf{K}\mathbf{A})^{T}$',
'e4343': [r'$\mathrm{kurt}(y) \;=\; E[y^{4}]-3\,, \qquad y=\mathbf{w}^{T}\mathbf{z}\,, \quad E[y^{2}]=1$',
          r'$J(y) \;\propto\; \left[\,E\,G(y)-E\,G(\nu)\,\right]^{2}\,, \qquad \nu\sim N(0,1)$'],
'e4344': [r'$E\left[\mathbf{z}\,g(\mathbf{w}^{T}\mathbf{z})\right]-\beta\,\mathbf{w}=\mathbf{0}\,, \qquad \beta=E\left[\mathbf{w}_0^{T}\mathbf{z}\,g(\mathbf{w}_0^{T}\mathbf{z})\right]$',
          r'$\mathbf{J}_F \;\approx\; \left(E[g^{\prime}(\mathbf{w}^{T}\mathbf{z})]-\beta\right)\mathbf{I}\,, \qquad \mathbf{w}-\mathbf{J}_F^{-1}\mathbf{F}\;\propto\;\mathbf{w}^{+}$'],
'e4345': r'$\mathbf{w}^{+} \;=\; E\left[\,\mathbf{z}\;g(\mathbf{w}^{T}\mathbf{z})\,\right]\;-\;E\left[\,g^{\prime}(\mathbf{w}^{T}\mathbf{z})\,\right]\mathbf{w}\,, \qquad \mathbf{w}\;\leftarrow\;\dfrac{\mathbf{w}^{+}}{\Vert\mathbf{w}^{+}\Vert}$',
'e4346': r'$\mathbf{W} \;\leftarrow\; (\mathbf{W}\mathbf{W}^{T})^{-1/2}\,\mathbf{W}$',
'e4347': [r'$\mathtt{components\_} \;=\; \mathbf{W}\mathbf{K}\,, \qquad \mathbf{S} \;=\; (\mathbf{X}-\mathbf{1}\boldsymbol{\mu}^{T})\,(\mathbf{W}\mathbf{K})^{T}$',
          r'$\mathbf{M} \;=\; (\mathbf{W}\mathbf{K})^{+}=\mathtt{mixing\_}\,, \qquad \mathbf{X} \;\approx\; \mathbf{S}\,\mathbf{M}^{T}+\mathbf{1}\boldsymbol{\mu}^{T}$'],
# 43.5  factor analysis
'e4351': r'$\mathbf{x} \;=\; \boldsymbol{\mu}+\mathbf{L}\,\mathbf{z}+\boldsymbol{\varepsilon}\,, \quad \mathbf{z}\sim N(\mathbf{0},\mathbf{I}_k)\,, \quad \boldsymbol{\varepsilon}\sim N(\mathbf{0},\boldsymbol{\Psi})\,, \quad \boldsymbol{\Psi}=\mathrm{diag}(\psi_1,\dots,\psi_p)$',
'e4352': r'$\boldsymbol{\Sigma} \;=\; \mathrm{Cov}(\mathbf{x}) \;=\; \mathbf{L}\mathbf{L}^{T}+\boldsymbol{\Psi}\,, \qquad \Sigma_{ii} \;=\; \sum_{j=1}^{k}L_{ij}^{2}+\psi_i$',
'e4353': r'$\ell(\mathbf{L},\boldsymbol{\Psi}) \;=\; -\dfrac{n}{2}\left[\,p\log 2\pi+\log\det\boldsymbol{\Sigma}+\mathrm{tr}\left(\boldsymbol{\Sigma}^{-1}\mathbf{S}\right)\right]\,, \qquad \mathbf{S}=\dfrac{1}{n}\,\mathbf{X}_c^{T}\mathbf{X}_c$',
'e4354': [r'$\boldsymbol{\Psi}^{-1/2}\,\mathbf{S}\,\boldsymbol{\Psi}^{-1/2} \;=\; \mathbf{V}\,\boldsymbol{\Theta}\,\mathbf{V}^{T}\,, \qquad \mathbf{L} \;=\; \boldsymbol{\Psi}^{1/2}\,\mathbf{V}_k\,\left(\max\{\boldsymbol{\Theta}_k-\mathbf{I},\,0\}\right)^{1/2}$',
          r'$\psi_i \;\leftarrow\; S_{ii}-\sum_{j}L_{ij}^{2}$'],
'e4355': r'$E[\mathbf{z}\,|\,\mathbf{x}] \;=\; \left(\mathbf{I}+\mathbf{L}^{T}\boldsymbol{\Psi}^{-1}\mathbf{L}\right)^{-1}\,\mathbf{L}^{T}\boldsymbol{\Psi}^{-1}\,(\mathbf{x}-\boldsymbol{\mu})$',
'e4356': [r'$\mathbf{L}\;\to\;\mathbf{L}\mathbf{R}\,,\ \ \mathbf{R}^{T}\mathbf{R}=\mathbf{I}\quad\Rightarrow\quad \boldsymbol{\Sigma}\ \ \mathrm{unchanged}$',
          r'$\mathrm{varimax:}\ \ \max_{\mathbf{R}}\ \sum_{j=1}^{k}\left[\dfrac{1}{p}\sum_{i}L_{ij}^{4}-\left(\dfrac{1}{p}\sum_{i}L_{ij}^{2}\right)^{2}\right]$'],
# 43.6  dictionary learning
'e4361': r'$\min_{\mathbf{W},\,\mathbf{H}}\ \ \dfrac{1}{2}\,\Vert\mathbf{X}-\mathbf{W}\mathbf{H}\Vert_F^{2}+\alpha\sum_{i,t}\vert w_{it}\vert \qquad \mathrm{subject\ to}\ \ \Vert\mathbf{h}_t\Vert_2\leq 1\,,\ \ t=1,\dots,k$',
'e4362': [r'$\min_{\mathbf{w}}\ \ \dfrac{1}{2}\,\Vert\mathbf{x}-\mathbf{H}^{T}\mathbf{w}\Vert^{2}+\alpha\,\Vert\mathbf{w}\Vert_1$',
          r'$\mathrm{threshold:}\ \ w_t=\mathrm{sign}(c_t)\,\max\{\vert c_t\vert-\alpha,\,0\}\,,\ \ \mathbf{c}=\mathbf{H}\mathbf{x}$'],
# 43.7  LDA
'e4371': [r'$\boldsymbol{\theta}_d\sim\mathrm{Dir}(\alpha\mathbf{1}_K)\,, \qquad z_{dn}\,|\,\boldsymbol{\theta}_d\sim\mathrm{Cat}(\boldsymbol{\theta}_d)$',
          r'$w_{dn}\,|\,z_{dn}=t\ \sim\ \mathrm{Cat}(\boldsymbol{\beta}_t)\,, \qquad \boldsymbol{\beta}_t\sim\mathrm{Dir}(\eta\mathbf{1}_V)$'],
'e4372': r'$\mathrm{Dir}(\boldsymbol{\theta}\,|\,\alpha) \;=\; \dfrac{\Gamma(K\alpha)}{\Gamma(\alpha)^{K}}\prod_{t=1}^{K}\theta_t^{\,\alpha-1}\,, \qquad E[\theta_t] \;=\; \dfrac{1}{K}\,, \qquad \theta_t\geq 0\,,\ \ \sum_t\theta_t=1$',
'e4373': [r'$\phi_{dwt} \;\propto\; \exp\left(E[\log\theta_{dt}]\right)\,\exp\left(E[\log\beta_{tw}]\right)\,, \qquad \gamma_{dt} \;=\; \alpha+\sum_{w}n_{dw}\,\phi_{dwt}$',
          r'$E[\log\theta_{dt}] \;=\; \psi(\gamma_{dt})-\psi\left(\sum_{s}\gamma_{ds}\right)$'],
'e4374': [r'$\lambda_{tw} \;=\; \eta+\sum_{d}n_{dw}\,\phi_{dwt}$',
          r'$\boldsymbol{\lambda}\;\leftarrow\;(1-\rho_{s})\,\boldsymbol{\lambda}+\rho_{s}\left(\eta+\dfrac{D}{\vert B\vert}\,\mathrm{stats}_B\right)\,, \qquad \rho_{s}=(\tau_0+s)^{-\kappa}$'],
'e4375': [r'$\hat{\theta}_{dt} \;=\; \gamma_{dt}\,/\sum_{s}\gamma_{ds}\,, \qquad \hat{\beta}_{tw} \;=\; \lambda_{tw}\,/\sum_{v}\lambda_{tv}$',
          r'$\mathrm{perplexity} \;=\; \exp\left(-\mathrm{bound}\,/\sum_{d,w}n_{dw}\right)$'],
}
mkeq(E,'.')
# by hand
matimg('m4381',[r'$\mathbf{X}\;=\;$',[['4','2'],['3','3'],['2','4']],r'$\;=\;$',[['2','0'],['1','1'],['0','2']],[['2','1'],['1','2']],r'$\;=\;\mathbf{W}_{\ast}\mathbf{H}_{\ast}$'],'.')
matimg('m4382',[r'$\mathbf{W}_0\;=\;$',[['2','1'],['1','1'],['1','2']],r'$,\qquad \mathbf{H}_0\;=\;$',[['1','1'],['1','1']]],'.')
matimg('m4383',[r'$\mathbf{X}\;=\;\mathbf{S}\mathbf{A}^{T}\;=\;$',[['1','1'],['1','-1'],['-1','-1'],['-1','1']],[['2','1'],['1','1']],r'$\;=\;$',[['3','2'],['1','0'],['-3','-2'],['-1','0']]],'.')
matimg('m4384',[r'$\mathbf{R}\;=\;$',[['1','0.72','0.54'],['0.72','1','0.48'],['0.54','0.48','1']]],'.')
print('eqs ok')
