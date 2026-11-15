# Equation and matrix images of Chapter 45. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq, matimg
E={
# 45.3 random projection
'e4531': r'$\mathbf{z}\;=\;\mathbf{R}\,\mathbf{x}\,,\qquad \mathbf{R}\in\mathbb{R}^{k\times p}\,,\qquad R_{jl}\;\sim\;\mathcal{N}\!\left(0,\,\dfrac{1}{k}\right)$',
'e4532': r'$q\;=\;\dfrac{\Vert\mathbf{R}(\mathbf{u}-\mathbf{v})\Vert^{2}}{\Vert\mathbf{u}-\mathbf{v}\Vert^{2}}\;=\;\dfrac{1}{k}\sum_{j=1}^{k}g_j^{2}\,,\qquad g_j\sim\mathcal{N}(0,1)\,,\qquad \mathrm{E}\,q=1\,,\quad \mathrm{Var}\,q=\dfrac{2}{k}$',
'e4533': [r'$P(q\geq 1+\varepsilon)\;\leq\;\left[(1+\varepsilon)\,e^{-\varepsilon}\right]^{k/2}\;\leq\;\exp\!\left[-\dfrac{k}{2}\left(\dfrac{\varepsilon^{2}}{2}-\dfrac{\varepsilon^{3}}{3}\right)\right]$',
          r'$P(q\leq 1-\varepsilon)\;\leq\;\left[(1-\varepsilon)\,e^{\varepsilon}\right]^{k/2}\;\leq\;\exp\!\left[-\dfrac{k}{2}\left(\dfrac{\varepsilon^{2}}{2}-\dfrac{\varepsilon^{3}}{3}\right)\right]$'],
'e4534': r'$k\;\geq\;\dfrac{4\,\ln n}{\varepsilon^{2}/2-\varepsilon^{3}/3}$',
'e4535': r'$R_{jl}\;=\;\sqrt{\dfrac{s}{k}}\;r_{jl}\,,\qquad P(r_{jl}=-1)=P(r_{jl}=+1)=\dfrac{1}{2s}\,,\qquad P(r_{jl}=0)=1-\dfrac{1}{s}$',
'e4536': r'$\mathrm{E}\,\Vert\mathbf{R}\mathbf{x}\Vert^{2}\;=\;\Vert\mathbf{x}\Vert^{2}\,,\qquad \mathrm{Var}\,\Vert\mathbf{R}\mathbf{x}\Vert^{2}\;=\;\dfrac{2\,\Vert\mathbf{x}\Vert^{4}+(s-3)\sum_{l=1}^{p}x_l^{4}}{k}$',
# 45.4 kernel approximation
'e4541': r'$k(\mathbf{x},\mathbf{z})\;\approx\;\hat{\varphi}(\mathbf{x})^{T}\hat{\varphi}(\mathbf{z})\,,\qquad \hat{\varphi}:\mathbb{R}^{p}\to\mathbb{R}^{D}\,,\qquad \mathbf{K}\;\approx\;\mathbf{Z}\mathbf{Z}^{T}\,,\quad \mathbf{Z}\in\mathbb{R}^{n\times D}$',
'e4542': r'$e^{-\gamma\Vert\boldsymbol{\delta}\Vert^{2}}\;=\;\int p(\boldsymbol{\omega})\,e^{\,i\boldsymbol{\omega}^{T}\boldsymbol{\delta}}\,d\boldsymbol{\omega}\;=\;\mathrm{E}\,\cos(\boldsymbol{\omega}^{T}\boldsymbol{\delta})\,,\qquad \boldsymbol{\omega}\sim\mathcal{N}(\mathbf{0},\,2\gamma\,\mathbf{I})$',
'e4543': [r'$2\cos(\boldsymbol{\omega}^{T}\mathbf{x}+b)\,\cos(\boldsymbol{\omega}^{T}\mathbf{z}+b)$',
          r'$=\;\cos\left(\boldsymbol{\omega}^{T}(\mathbf{x}-\mathbf{z})\right)+\cos\left(\boldsymbol{\omega}^{T}(\mathbf{x}+\mathbf{z})+2b\right)$'],
'e4544': r'$\hat{\varphi}(\mathbf{x})\;=\;\sqrt{\dfrac{2}{D}}\,\left[\cos(\boldsymbol{\omega}_1^{T}\mathbf{x}+b_1)\,,\ \ldots\,,\ \cos(\boldsymbol{\omega}_D^{T}\mathbf{x}+b_D)\right]^{T}$',
'e4545': r'$\mathrm{Var}\!\left[\hat{\varphi}(\mathbf{x})^{T}\hat{\varphi}(\mathbf{z})\right]\;=\;\dfrac{1}{D}\left(1+\dfrac{\kappa^{4}}{2}-\kappa^{2}\right)\;\leq\;\dfrac{1}{D}\,,\qquad \kappa=k(\mathbf{x},\mathbf{z})$',
'e4546': r'$\hat{\mathbf{K}}\;=\;\mathbf{C}\,\mathbf{W}^{-1}\mathbf{C}^{T}\,,\qquad \mathbf{C}=\mathbf{K}_{:,I}\in\mathbb{R}^{n\times m}\,,\qquad \mathbf{W}=\mathbf{K}_{I,I}\in\mathbb{R}^{m\times m}$',
'e4547': [r'$\mathbf{W}\,\boldsymbol{\beta}(\mathbf{x})\;=\;\mathbf{k}_I(\mathbf{x})\,,\qquad \mathbf{k}_I(\mathbf{x})=\left[k(\mathbf{x},\mathbf{x}_j)\right]_{j\in I}$',
          r'$\hat{k}(\mathbf{x},\mathbf{z})\;=\;\boldsymbol{\beta}(\mathbf{x})^{T}\mathbf{W}\,\boldsymbol{\beta}(\mathbf{z})\;=\;\mathbf{k}_I(\mathbf{x})^{T}\,\mathbf{W}^{-1}\,\mathbf{k}_I(\mathbf{z})$'],
'e4548': r'$\hat{\varphi}(\mathbf{x})\;=\;\mathbf{W}^{-1/2}\,\mathbf{k}_I(\mathbf{x})\,,\qquad \mathbf{W}=\mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^{T}\;\Rightarrow\;\mathbf{W}^{-1/2}=\mathbf{U}\boldsymbol{\Lambda}^{-1/2}\mathbf{U}^{T}$',
}
mkeq(E,'.')
