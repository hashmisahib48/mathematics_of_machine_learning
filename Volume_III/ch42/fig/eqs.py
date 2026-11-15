# Equation and matrix images of Chapter 42. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq, matimg
import matplotlib.pyplot as plt
X=r'\mathbf{X}'; Xc=r'\mathbf{X}_c'; S=r'\mathbf{S}'
E={
# 42.3
'e4231': rf'$\mathbf{{S}} \;=\; \dfrac{{1}}{{n-1}}\,\mathbf{{X}}_c^{{T}}\mathbf{{X}}_c\,, \qquad \mathbf{{X}}_c \;=\; \mathbf{{X}}-\mathbf{{1}}\boldsymbol{{\mu}}^{{T}}\,, \qquad \boldsymbol{{\mu}} \;=\; \dfrac{{1}}{{n}}\,\mathbf{{X}}^{{T}}\mathbf{{1}}$',
'e4232': r'$z_i \;=\; \mathbf{w}^{T}(\mathbf{x}_i-\boldsymbol{\mu})\,, \qquad \mathrm{Var}(z) \;=\; \dfrac{1}{n-1}\sum_{i=1}^{n} z_i^{2} \;=\; \mathbf{w}^{T}\mathbf{S}\,\mathbf{w}$',
'e4233': r'$\mathcal{L}(\mathbf{w},\lambda) \;=\; \mathbf{w}^{T}\mathbf{S}\,\mathbf{w} \;-\; \lambda\,(\mathbf{w}^{T}\mathbf{w}-1)$',
'e4234': r'$\dfrac{\partial\mathcal{L}}{\partial\mathbf{w}} \;=\; 2\,\mathbf{S}\mathbf{w}-2\lambda\,\mathbf{w} \;=\; \mathbf{0} \quad\Longrightarrow\quad \mathbf{S}\mathbf{w} \;=\; \lambda\,\mathbf{w}\,, \qquad \mathbf{w}^{T}\mathbf{S}\mathbf{w} \;=\; \lambda$',
'e4235': r'$\mathbf{S} \;=\; \mathbf{W}\boldsymbol{\Lambda}\mathbf{W}^{T}\,, \qquad \mathrm{tr}\,\mathbf{S} \;=\; \sum_{j=1}^{p} S_{jj} \;=\; \sum_{j=1}^{p}\lambda_j$',
'e4236': r'$\sum_{i=1}^{n}\Vert\mathbf{x}_i-\hat{\mathbf{x}}_i\Vert^{2} \;=\; \sum_{i=1}^{n}\,\sum_{j>k} z_{ij}^{2} \;=\; (n-1)\sum_{j=k+1}^{p}\lambda_j$',
# 42.4
'e4241': r'$\mathbf{X}_c \;=\; \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^{T}\,, \qquad \mathbf{S} \;=\; \dfrac{1}{n-1}\,\mathbf{V}\boldsymbol{\Sigma}^{2}\mathbf{V}^{T}$',
'e4242': r'$\mathbf{w}_j \;=\; \mathbf{v}_j\,, \qquad \lambda_j \;=\; \dfrac{\sigma_j^{2}}{n-1}\,, \qquad r_j \;=\; \dfrac{\lambda_j}{\lambda_1+\cdots+\lambda_p}$',
'e4243': r'$\mathbf{Z} \;=\; \mathbf{X}_c\mathbf{V}_k \;=\; \mathbf{U}_k\boldsymbol{\Sigma}_k\,, \qquad \hat{\mathbf{X}} \;=\; \mathbf{Z}\mathbf{V}_k^{T}+\mathbf{1}\boldsymbol{\mu}^{T}$',
'e4244': r'$\mathbf{Z}_{w} \;=\; \mathbf{Z}\,\boldsymbol{\Lambda}_k^{-1/2} \;=\; \sqrt{n-1}\;\mathbf{U}_k\,, \qquad \hat{\mathbf{X}} \;=\; \mathbf{Z}_{w}\boldsymbol{\Lambda}_k^{1/2}\mathbf{V}_k^{T}+\mathbf{1}\boldsymbol{\mu}^{T}$',
# 42.5
'e4251': r'$k(f) \;=\; \min\left\{\,k\,:\ \sum_{j=1}^{k} r_j \;>\; f\,\right\}$',
'e4252': r'$\sigma_k^{2} \;=\; \dfrac{1}{p-k}\;\sum_{j=k+1}^{p}\lambda_j\,, \qquad \hat{\boldsymbol{\Sigma}}_k \;=\; \sum_{j=1}^{k}(\lambda_j-\sigma_k^{2})\,\mathbf{w}_j\mathbf{w}_j^{T}+\sigma_k^{2}\,\mathbf{I}$',
# 42.6
'e4262': r'$\det(\mathbf{S}-\lambda\mathbf{I}) \;=\; (5-\lambda)(2-\lambda)-4 \;=\; \lambda^{2}-7\lambda+6 \;=\; (\lambda-6)(\lambda-1)$',
'e4263': [r'$\mathbf{w}_1 \;=\; \dfrac{1}{\sqrt{5}}\,(2,\;1)^{T} \;=\; (0.8944,\;0.4472)^{T}$', r'$\mathbf{w}_2 \;=\; \dfrac{1}{\sqrt{5}}\,(-1,\;2)^{T} \;=\; (-0.4472,\;0.8944)^{T}$'],
'e4264': r'$z_{i1} \;=\; \dfrac{2\,x_{ci}+y_{ci}}{\sqrt{5}}\,, \qquad z_{i2} \;=\; \dfrac{-x_{ci}+2\,y_{ci}}{\sqrt{5}}$',
# 42.11
'e42111': r'$\tilde{\mathbf{K}}\,\mathbf{a}_m \;=\; \mu_m\,\mathbf{a}_m\,, \qquad \mu_m \;=\; (n-1)\,\lambda_m\,, \qquad \tilde{K}_{il} \;=\; \tilde{\varphi}(\mathbf{x}_i)^{T}\tilde{\varphi}(\mathbf{x}_l)$',
'e42112': r'$\tilde{\mathbf{K}} \;=\; \mathbf{J}\,\mathbf{K}\,\mathbf{J}\,, \qquad \mathbf{J} \;=\; \mathbf{I}-\dfrac{1}{n}\,\mathbf{1}\mathbf{1}^{T}$',
'e42113': r'$z_{im} \;=\; \sqrt{\mu_m}\;\alpha_{im}\,, \qquad z_m(\mathbf{x}) \;=\; \dfrac{1}{\sqrt{\mu_m}}\sum_{i=1}^{n}\alpha_{im}\,\tilde{k}(\mathbf{x},\mathbf{x}_i)$',
'e42114': r'$\hat{\mathbf{x}}(\mathbf{z}) \;=\; \mathbf{k}(\mathbf{z},\mathbf{Z})\,\left(\mathbf{K}_{Z}+\alpha\,\mathbf{I}\right)^{-1}\mathbf{X}$',
# 42.12
'e42121': r'$\min_{\mathbf{A},\,\mathbf{C}}\ \ \dfrac{1}{2}\,\Vert\mathbf{X}_c-\mathbf{A}\mathbf{C}^{T}\Vert_F^{2} \;+\; \alpha\sum_{j,m}\vert c_{jm}\vert \qquad \mathrm{subject\ to}\ \ \Vert\mathbf{a}_m\Vert_2=1\,,\ \ m=1,\dots,k$',
# 42.13
'e42131': r'$\mathbf{X} \;\approx\; \mathbf{U}_k\boldsymbol{\Sigma}_k\mathbf{V}_k^{T}\,, \qquad \mathbf{Z} \;=\; \mathbf{X}\mathbf{V}_k \;=\; \mathbf{U}_k\boldsymbol{\Sigma}_k$',
}
mkeq(E,'.')
matimg('m4261',[r'$\mathbf{S}\;=\;\dfrac{1}{4}$',[['20','8'],['8','8']],r'$\;=\;$',[['5','2'],['2','2']]],'.')
matimg('m4265',[r'$\mathbf{K}\;=\;$',[['16','4','4','16'],['4','1','1','4'],['4','1','1','4'],['16','4','4','16']],r'$\,,\quad \tilde{\mathbf{K}}\;=\;\dfrac{9}{4}$',[['1','-1','-1','1'],['-1','1','1','-1'],['-1','1','1','-1'],['1','-1','-1','1']]],'.',rh=0.19)
print('eqs ok')
