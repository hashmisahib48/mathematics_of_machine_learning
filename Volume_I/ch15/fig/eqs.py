# Equation and matrix images of Chapter 15. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq, matimg
T=r'^{\mathrm{T}}'; w=r'\mathbf{w}'; y=r'\mathbf{y}'; Phi=r'\mathbf{\Phi}'
E={
'e1521': r'$\hat{y} \;=\; b + \sum_{k=1}^{m} w_k\,\varphi_k(x) \;=\; b + w_1\varphi_1(x) + w_2\varphi_2(x) + \cdots + w_m\varphi_m(x)$',
'e1522': rf'$J({w},b) \;=\; \Vert {y} - b\mathbf{{1}} - {Phi}{w} \Vert^2 \;+\; \alpha\,\Vert {w} \Vert^2$',
'e1523': rf'$\hat{{{w}}} \;=\; ({Phi}_c{T}{Phi}_c + \alpha\mathbf{{I}})^{{-1}}\,{Phi}_c{T}\,{y}_c\,, \qquad b \;=\; \bar{{y}} - \bar{{\boldsymbol{{\varphi}}}}{T}\hat{{{w}}}$'.replace(r'\boldsymbol{\varphi}',r'\mathbf{\varphi}'),
'e1531': r'$x_1^{a_1}\,x_2^{a_2}\cdots x_p^{a_p} \quad\mathrm{with}\quad a_1+a_2+\cdots+a_p \;\leq\; d\,, \qquad a_j \in \{0,1,2,\ldots\}$',
'e1532': r'$m \;=\; \binom{p+d}{d} \quad\mathrm{(with\ bias\ column)}, \qquad m \;=\; \sum_{k=0}^{\min(d,\,p)} \binom{p}{k} \quad\mathrm{(interaction\_only)}$',
'e1533': r'$\hat{y} \;=\; b + w_1x_1 + w_2x_2 + w_{11}x_1^{2} + w_{12}\,x_1x_2 + w_{22}x_2^{2}$',
'e1541': r'$B_{j,0}(x) \;=\; \begin{cases}1 & t_j \leq x < t_{j+1}\\ 0 & \mathrm{otherwise}\end{cases}'.replace(r'\begin{cases}1 & t_j \leq x < t_{j+1}\\ 0 & \mathrm{otherwise}\end{cases}',r'\left\{\,1\ \mathrm{if}\ t_j\leq x<t_{j+1},\ \ 0\ \mathrm{otherwise}\right.')+'$',
'e1542': r'$B_{j,k}(x) \;=\; \dfrac{x-t_j}{t_{j+k}-t_j}\,B_{j,k-1}(x) \;+\; \dfrac{t_{j+k+1}-x}{t_{j+k+1}-t_{j+1}}\,B_{j+1,k-1}(x)$',
'e1543': r'$t_j \;=\; x_{\min} + j\,h\,, \quad j=-d,\ldots,n_{\mathrm{knots}}-1+d\,, \qquad h \;=\; \dfrac{x_{\max}-x_{\min}}{n_{\mathrm{knots}}-1}$',
'e1544': r'$f(x) \;=\; \sum_{j=1}^{m} w_j\,B_j(x)\,, \qquad m = n_{\mathrm{knots}}+d-1\,, \qquad \sum_{j=1}^{m} B_j(x) \;=\; 1$',
'e1551': r'$\hat{y} \;=\; \bar{y} + c_1\,u + c_2\,q\,, \qquad u = x-\bar{x}\,, \quad q = u^2-s\,, \quad s = \dfrac{1}{n}\sum u_i^2$',
'e1552': r'$c_1 \;=\; \dfrac{\sum u_i\,y_i}{\sum u_i^{2}}\,, \qquad c_2 \;=\; \dfrac{\sum q_i\,y_i}{\sum q_i^{2}}$',
'e1553': r'$\hat{y} \;=\; 10 + 5.2\,(x-3) + \left[(x-3)^2-2\right] \;=\; 1.4 - 0.8\,x + 1.0\,x^2$',
'e1554': r'$B_{0,2}(1.5) \;=\; \dfrac{1.5-0}{2-0}\times 0.5 \;+\; \dfrac{3-1.5}{3-1}\times 0.5 \;=\; 0.375+0.375 \;=\; 0.75$',
}
mkeq(E,'.')
M=lambda rows:[[str(v) for v in r] for r in rows]
matimg('m1551',[rf'${Phi}=$',M([[1,0,0],[0.5,0.5,0],[0,1,0],[0,0.5,0.5],[0,0,1]]),r'$,\quad \mathbf{y}=$',M([[2],[3],[8],[15],[22]])])
matimg('m1552',[rf'${Phi}{T}{Phi}=$',M([[1.25,0.25,0],[0.25,1.5,0.25],[0,0.25,1.25]]),rf'$,\quad {Phi}{T}{y}=$',M([[3.5],[17],[29.5]])])
print('eqs ok')
