# Equation images of Chapter 25. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq
E={
'e2531': r'$c_i \;=\; \#\{\,j:\ i_j = i\,\}\,, \qquad \sum_{i=1}^{n} c_i \;=\; n$',
'e2532': r'$\ln P(c_i=0) \;=\; n\ln\left(1-\dfrac{1}{n}\right) \;=\; -1-\dfrac{1}{2n}-\dfrac{1}{3n^{2}}-\cdots$',
'e2533': r'$P(c_i=0) \;=\; \left(1-\dfrac{1}{n}\right)^{n} \;\longrightarrow\; e^{-1} \;=\; 0.3679 \qquad (n\to\infty)$',
'e2534': r'$P(c_i=k) \;=\; \dfrac{n!}{k!\,(n-k)!}\left(\dfrac{1}{n}\right)^{k}\left(1-\dfrac{1}{n}\right)^{n-k} \;\longrightarrow\; \dfrac{e^{-1}}{k!}$',
'e2535': r'$\bar{f}(\mathbf{x}) \;=\; \dfrac{1}{B}\sum_{b=1}^{B} f_b(\mathbf{x})$',
'e2536': r'$\mathrm{Var}\,[\bar{f}] \;=\; \dfrac{1}{B^{2}}\left[\ \sum_{b=1}^{B}\mathrm{Var}\,[f_b] \;+\; \sum_{b\neq b^{\prime}}\mathrm{Cov}\,(f_b,f_{b^{\prime}})\right]$',
'e2537': r'$\mathrm{Var}\,[\bar{f}] \;=\; \dfrac{1}{B^{2}}\left[\,B\,\sigma^{2} \;+\; B\,(B-1)\,\rho\,\sigma^{2}\,\right]$',
'e2538': r'$\mathrm{Var}\,[\bar{f}] \;=\; \rho\,\sigma^{2} \;+\; \dfrac{1-\rho}{B}\,\sigma^{2}$',
'e2539': r'$\hat{P}(k\,|\,\mathbf{x}) \;=\; \dfrac{1}{B}\sum_{b=1}^{B} p_{bk}(\mathbf{x})\,, \qquad \hat{y}(\mathbf{x}) \;=\; \arg\max_{k}\ \hat{P}(k\,|\,\mathbf{x})$',
'e2541': r'$m \;=\; \max\left(1,\ \lfloor\sqrt{p}\,\rfloor\right)$',
'e2542': r'$s \;\sim\; \mathrm{Uniform}\left(v_{\min},\,v_{\max}\right)\,, \qquad v_{\min}=\min_{i\in t} x_{ij}\,, \quad v_{\max}=\max_{i\in t} x_{ij}$',
'e2551': r'$\mathcal{B}_i \;=\; \{\,b:\ c_{bi}=0\,\}\,, \qquad E\,|\mathcal{B}_i| \;=\; B\,e^{-1}$',
'e2552': r'$\hat{P}_{\mathrm{oob}}(k\,|\,\mathbf{x}_i) \;=\; \dfrac{1}{|\mathcal{B}_i|}\sum_{b\in\mathcal{B}_i} p_{bk}(\mathbf{x}_i)\,, \qquad \hat{y}_i^{\mathrm{oob}} \;=\; \arg\max_{k}\ \hat{P}_{\mathrm{oob}}(k\,|\,\mathbf{x}_i)$',
'e2553': r'$\mathrm{acc}_{\mathrm{oob}} \;=\; \dfrac{1}{n}\sum_{i=1}^{n}\mathbf{1}\left[\,\hat{y}_i^{\mathrm{oob}}=y_i\,\right]$',
'e2591': r'$\mathrm{imp}(j) \;=\; \dfrac{1}{B}\sum_{b=1}^{B}\ \mathrm{imp}_b(j)$',
'e2592': r'$\phi_{b\ell}(\mathbf{x}) \;=\; \mathbf{1}\left[\,\mathbf{x}\ \mathrm{reaches\ leaf}\ \ell\ \mathrm{of\ tree}\ b\,\right]\,, \qquad \sum_{\ell}\phi_{b\ell}(\mathbf{x})=1$',
'e2561': r'$G_{\mathrm{split}} \;=\; \dfrac{4}{6}\left[1-\left(\dfrac{3}{4}\right)^{2}-\left(\dfrac{1}{4}\right)^{2}\right]+\dfrac{2}{6}\cdot 0 \;=\; \dfrac{4}{6}\times 0.375 \;=\; 0.25$',
'e2562': r'$\bar{f}(45) \;=\; \dfrac{0.25+1+1}{3} \;=\; 0.75$',
'e2563': r'$\mathrm{Var}\,[\bar{f}] \;=\; \dfrac{1}{9}\,\left(3\times 4 \;+\; 6\times 1\right) \;=\; 2\,, \qquad \rho\sigma^{2}+\dfrac{(1-\rho)\sigma^{2}}{B} \;=\; 1+\dfrac{3}{3} \;=\; 2$',
}
mkeq(E,'.')
print('eqs ok')
