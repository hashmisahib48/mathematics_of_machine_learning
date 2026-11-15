# Equation images of Chapter 44. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq, matimg
E={
'e4431': r'$d_{ij}^{2} \;=\; \Vert\mathbf{z}_i-\mathbf{z}_j\Vert^{2} \;=\; b_{ii}+b_{jj}-2\,b_{ij}\,, \qquad \mathbf{B} \;=\; \mathbf{Z}\mathbf{Z}^{T}$',
'e4432': r'$\mathbf{B} \;=\; -\dfrac{1}{2}\,\mathbf{J}\,\mathbf{D}^{(2)}\,\mathbf{J}\,, \qquad \mathbf{J} \;=\; \mathbf{I}-\dfrac{1}{n}\,\mathbf{1}\mathbf{1}^{T}\,, \qquad D^{(2)}_{ij} \;=\; d_{ij}^{2}$',
'e4433': r'$\mathbf{B} \;=\; \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{T}\,, \qquad \mathbf{Z} \;=\; \mathbf{V}_k\,\boldsymbol{\Lambda}_k^{1/2}$',
'e4434': r'$\mathbf{B} \;=\; \mathbf{X}_c\mathbf{X}_c^{T}\,, \qquad \mu_j \;=\; \sigma_j^{2} \;=\; (n-1)\,\lambda_j\,, \qquad \mathbf{Z} \;=\; \mathbf{U}_k\boldsymbol{\Sigma}_k$',
'e4435': r'$d^{G}_{ij} \;=\; \min_{\mathrm{paths}\ i=v_0,\,v_1,\dots,v_s=j}\ \ \sum_{r=1}^{s}\Vert\mathbf{x}_{v_{r-1}}-\mathbf{x}_{v_r}\Vert\,, \qquad \mathbf{B} \;=\; -\dfrac{1}{2}\,\mathbf{J}\,\mathbf{D}^{G\,(2)}\mathbf{J}$',
'e4441': r'$\sigma(\mathbf{Z}) \;=\; \sum_{i<j}\left(d_{ij}-\Vert\mathbf{z}_i-\mathbf{z}_j\Vert\right)^{2}$',
'e4442': r'$\sigma_1(\mathbf{Z}) \;=\; \sqrt{\dfrac{\sum_{i<j}\left(d_{ij}-\Vert\mathbf{z}_i-\mathbf{z}_j\Vert\right)^{2}}{\sum_{i<j}\Vert\mathbf{z}_i-\mathbf{z}_j\Vert^{2}}}$',
'e4443': r'$\sigma(\mathbf{Z}) \;\leq\; \sum_{i<j}d_{ij}^{2}\;-\;2\,\mathrm{tr}\!\left(\mathbf{Z}^{T}\mathbf{B}(\mathbf{W})\mathbf{W}\right)\;+\;n\,\mathrm{tr}\!\left(\mathbf{Z}^{T}\mathbf{Z}\right)\,, \quad \mathrm{equality\ at}\ \mathbf{Z}=\mathbf{W}$',
'e4444': r'$B(\mathbf{W})_{ij} \;=\; -\dfrac{d_{ij}}{\Vert\mathbf{w}_i-\mathbf{w}_j\Vert}\ \ (i\neq j)\,, \qquad B(\mathbf{W})_{ii} \;=\; -\sum_{j\neq i}B(\mathbf{W})_{ij}$',
'e4445': r'$\mathbf{Z}^{+} \;=\; \dfrac{1}{n}\,\mathbf{B}(\mathbf{Z})\,\mathbf{Z}$',
'e4451': r'$\varepsilon_i(\mathbf{w}) \;=\; \left\Vert\mathbf{x}_i-\sum_{j}w_j\,\mathbf{x}_j\right\Vert^{2} \;=\; \mathbf{w}^{T}\mathbf{G}\,\mathbf{w}\,, \qquad G_{jl} \;=\; (\mathbf{x}_i-\mathbf{x}_j)^{T}(\mathbf{x}_i-\mathbf{x}_l)$',
'e4452': r'$\mathcal{L}(\mathbf{w},\lambda) \;=\; \mathbf{w}^{T}\mathbf{G}\mathbf{w}-\lambda\,(\mathbf{1}^{T}\mathbf{w}-1)\,, \qquad \dfrac{\partial\mathcal{L}}{\partial\mathbf{w}} \;=\; 2\,\mathbf{G}\mathbf{w}-\lambda\,\mathbf{1} \;=\; \mathbf{0}$',
'e4453': r'$\mathbf{w} \;=\; \dfrac{\mathbf{G}^{-1}\mathbf{1}}{\mathbf{1}^{T}\mathbf{G}^{-1}\mathbf{1}}$',
'e4454': r'$\mathbf{G} \;\leftarrow\; \mathbf{G}+r\,\mathbf{I}\,, \qquad r \;=\; \mathtt{reg}\cdot\mathrm{tr}\,\mathbf{G}$',
'e4455': r'$\Phi(\mathbf{Z}) \;=\; \sum_{i}\left\Vert\mathbf{z}_i-\sum_{j}W_{ij}\,\mathbf{z}_j\right\Vert^{2} \;=\; \mathrm{tr}\!\left(\mathbf{Z}^{T}\mathbf{M}\,\mathbf{Z}\right)\,, \qquad \mathbf{M} \;=\; (\mathbf{I}-\mathbf{W})^{T}(\mathbf{I}-\mathbf{W})$',
'e4461': r'$\dfrac{1}{2}\sum_{i,j}W_{ij}\Vert\mathbf{z}_i-\mathbf{z}_j\Vert^{2} \;=\; \mathrm{tr}\!\left(\mathbf{Z}^{T}\mathbf{L}\,\mathbf{Z}\right)\,, \quad \mathbf{L}=\mathbf{D}-\mathbf{W}\,, \quad \mathbf{Z}^{T}\mathbf{D}\,\mathbf{Z}=\mathbf{I}$',
'e4462': r'$\mathbf{L}\,\mathbf{z} \;=\; \lambda\,\mathbf{D}\,\mathbf{z}\,, \qquad \mathbf{z} \;=\; \mathbf{D}^{-1/2}\,\mathbf{u}\,, \qquad \mathbf{D}^{-1/2}\mathbf{L}\,\mathbf{D}^{-1/2}\,\mathbf{u} \;=\; \lambda\,\mathbf{u}$',
'e4471': r'$p_{j\mid i} \;=\; \dfrac{\exp\left(-\Vert\mathbf{x}_i-\mathbf{x}_j\Vert^{2}/2\sigma_i^{2}\right)}{\sum_{l\neq i}\exp\left(-\Vert\mathbf{x}_i-\mathbf{x}_l\Vert^{2}/2\sigma_i^{2}\right)}\,, \qquad p_{i\mid i}=0$',
'e4472': r'$\mathrm{Perp}(P_i) \;=\; \exp\left(-\sum_{j}p_{j\mid i}\ln p_{j\mid i}\right)$',
'e4473': r'$p_{ij} \;=\; \dfrac{p_{j\mid i}+p_{i\mid j}}{2n}\,, \qquad \sum_{i\neq j}p_{ij}=1$',
'e4474': r'$q_{ij} \;=\; \dfrac{\left(1+\Vert\mathbf{z}_i-\mathbf{z}_j\Vert^{2}\right)^{-1}}{\sum_{k\neq l}\left(1+\Vert\mathbf{z}_k-\mathbf{z}_l\Vert^{2}\right)^{-1}}$',
'e4475': r'$C(\mathbf{Z}) \;=\; \mathrm{KL}(P\,\Vert\,Q) \;=\; \sum_{i\neq j}p_{ij}\,\ln\dfrac{p_{ij}}{q_{ij}}$',
'e4476': r'$\dfrac{\partial C}{\partial\mathbf{z}_i} \;=\; 4\sum_{j}\left(p_{ij}-q_{ij}\right)\dfrac{\mathbf{z}_i-\mathbf{z}_j}{1+\Vert\mathbf{z}_i-\mathbf{z}_j\Vert^{2}}$',
'e4491': r'$T(k) \;=\; 1-\dfrac{2}{n\,k\,(2n-3k-1)}\sum_{i=1}^{n}\ \sum_{j\in U_i(k)}\left(r(i,j)-k\right)$',
}
mkeq(E,'.')
print('eqs ok')
