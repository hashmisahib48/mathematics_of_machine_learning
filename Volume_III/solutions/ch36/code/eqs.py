# Equation images of the Chapter 36 solutions. Run: python3 code/eqs.py  (writes sol/ch36/s36*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 36A.1
's36a1_1': r'$J \;=\; (0-1)^2+(2-1)^2+(6-10)^2+(12-10)^2 \;=\; 1+1+16+4 \;=\; 22$',
's36a1_2': r'$\mu_1=\dfrac{0+2}{2}=1\,,\qquad \mu_2=\dfrac{6+12}{2}=9$',
's36a1_3': r'$J \;=\; (0-1)^2+(2-1)^2+(6-9)^2+(12-9)^2 \;=\; 1+1+9+9 \;=\; 20$',
's36a1_4': r'$(6-10)^2+(12-10)^2 \;=\; \left[(6-9)^2+(12-9)^2\right] + 2\,(9-10)^2 \quad\Longleftrightarrow\quad 20 \;=\; 18+2$',
# ---- 36A.2
's36a2_1': r'$\bar{x} \;=\; \dfrac{1+3+8+10}{4} \;=\; \dfrac{22}{4} \;=\; 5.5$',
's36a2_2': r'$\mathrm{SST} \;=\; (-4.5)^2+(-2.5)^2+(2.5)^2+(4.5)^2 \;=\; 20.25+6.25+6.25+20.25 \;=\; 53$',
's36a2_3': r'$\mu_1=\dfrac{1+3}{2}=2\,,\qquad \mu_2=\dfrac{8+10}{2}=9$',
's36a2_7': r'$\mathrm{SSW} \;=\; (1-2)^2+(3-2)^2+(8-9)^2+(10-9)^2 \;=\; 1+1+1+1 \;=\; 4$',
's36a2_4': r'$\mathrm{SSB} \;=\; 2\,(2-5.5)^2+2\,(9-5.5)^2 \;=\; 2\times12.25+2\times12.25 \;=\; 49$',
's36a2_5': r'$\mathrm{SSW}+\mathrm{SSB} \;=\; 4+49 \;=\; 53 \;=\; \mathrm{SST}$',
's36a2_6': r'$\dfrac{1}{2\,n_1}\sum_{i,j\in C_1}(x_i-x_j)^2 \;=\; \dfrac{1}{4}\,\left[\,0+(1-3)^2+(3-1)^2+0\,\right] \;=\; \dfrac{8}{4} \;=\; 2$',
# ---- 36A.3
's36a3_1': r'$D(x)^2:\qquad 0^2=0\,,\quad 1^2=1\,,\quad 2^2=4\,,\quad 10^2=100$',
's36a3_4': r'$\sum_{j} D(x_j)^2 \;=\; 0+1+4+100 \;=\; 105$',
's36a3_2': r'$P(1)=\dfrac{1}{105}\,,\quad P(2)=\dfrac{4}{105}\,,\quad P(10)=\dfrac{100}{105}\,,\qquad P(0)=0$',
's36a3_3': r'$\dfrac{P(10)}{1/3} \;=\; \dfrac{100/105}{1/3} \;=\; \dfrac{300}{105} \;\approx\; 2.86$',
# ---- 36C.1, 36C.2
's36c1_1': r'$\mathbf{u}^{T}\mathbf{x} \;\leq\; b\,,\qquad \Vert\mathbf{u}\Vert=1$',
's36c1_2': r'$-1.12 \;\leq\; \mathbf{u}^{T}\mathbf{x} \;\leq\; 1.12 \;\;\mathrm{(outer\ ring)}\,,\qquad -0.53 \;\leq\; \mathbf{u}^{T}\mathbf{x} \;\leq\; 0.53 \;\;\mathrm{(inner\ ring)}$',
's36c2_1': r'$J(K) \;\approx\; \dfrac{5}{18\sqrt{3}}\ \dfrac{n\,A}{K} \;\approx\; 0.160\ \dfrac{n\,A}{K}$',
# ---- questions
's36q1_1': r'$\boldsymbol{\mu} \;=\; \dfrac{\sum_i \mathbf{x}_i\,/\,\Vert\mathbf{x}_i-\boldsymbol{\mu}\Vert}{\sum_i 1\,/\,\Vert\mathbf{x}_i-\boldsymbol{\mu}\Vert}$',
's36q2_1': r'$\sum_{i\in C}\Vert\mathbf{x}_i-\mathbf{s}\Vert^2 \;=\; \sum_{i\in C}\Vert\mathbf{x}_i-\bar{\mathbf{x}}_C\Vert^2 \;+\; n_C\,\Vert\bar{\mathbf{x}}_C-\mathbf{s}\Vert^2$',
}
mkeq(E, '.'); print('solution equations ok:', len(E))
