# Equation images of the Chapter 35 solutions. Run: python3 code/eqs.py  (writes sol/ch35/s35*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq, matimg
T = r'^{T}'
E = {
# ---- 35A.1  first component
's35a1_1': [r'$\mathbf{x}_1^{T}\mathbf{y} = (-3)(-1)+(-1)(0)+(1)(1)+(3)(0) = 4\,,$',
            r'$\mathbf{x}_2^{T}\mathbf{y} = (-1)(-1)+(-2)(0)+(2)(1)+(1)(0) = 3$'],
's35a1_2': r'$\mathbf{w}_1 \;=\; \dfrac{(4,\ 3)}{\sqrt{4^2+3^2}} \;=\; \dfrac{(4,\ 3)}{5} \;=\; (0.8,\ 0.6)$',
's35a1_3': [r'$t_{i1} = 0.8\,x_{i1}+0.6\,x_{i2}:\qquad \mathbf{t}_1 = (-3,\ -2,\ 2,\ 3)$',
            r'$\mathbf{t}_1^{T}\mathbf{t}_1 = 9+4+4+9 = 26$'],
's35a1_4': [r'$\mathbf{p}_1 = \dfrac{\mathbf{X}^{T}\mathbf{t}_1}{26} = \dfrac{(22,\ 14)}{26} = (0.8462,\ 0.5385)\,,$',
            r'$q_1 = \dfrac{\mathbf{y}^{T}\mathbf{t}_1}{26} = \dfrac{3+0+2+0}{26} = \dfrac{5}{26} = 0.1923$'],
's35a1_5': r'$\mathbf{B}_1 \;=\; q_1\,\mathbf{w}_1 \;=\; \dfrac{5}{26}\,(0.8,\ 0.6) \;=\; \dfrac{(4,\ 3)}{26} \;=\; (0.1538,\ 0.1154)$',
's35a1_6': [r'$\mathbf{y}_1 \;=\; \mathbf{y}-q_1\mathbf{t}_1 \;=\; \dfrac{1}{26}\,(-11,\ 10,\ 16,\ -15)$',r'$=\; (-0.4231,\ 0.3846,\ 0.6154,\ -0.5769)$'],
's35a1_7': r'$\mathbf{X}_1^{T}\mathbf{y}_1 = (-0.2308,\ 0.3077)\,,\quad \mathbf{w}_2 = (-0.6,\ 0.8)\,,\quad \mathbf{t}_2^{T}\mathbf{t}_2 = 3.846\,,\quad q_2 = 0.1$',
's35a1_8': [r'$\mathbf{p}_2^{T}\mathbf{w}_1 = 0.8462\cdot\ldots$'],
's35a1_9': [r'$\mathbf{r}_1 = \mathbf{w}_1 = (0.8,\ 0.6)\,,\qquad \mathbf{r}_2 = \mathbf{w}_2 + \dfrac{1}{13}\mathbf{w}_1 = (-0.5385,\ 0.8462)$'],
's35a1_10': r'$\mathbf{B} \;=\; q_1\mathbf{r}_1+q_2\mathbf{r}_2 \;=\; (0.1538-0.0538,\ \ 0.1154+0.0846) \;=\; (0.1,\ 0.2)$',
's35a1_11': r'$(\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{y} \;=\; \dfrac{1}{100}\,\left(10\cdot 4-10\cdot 3,\ \ -10\cdot 4+20\cdot 3\right) \;=\; (0.1,\ 0.2)$',
# ---- 35A.2
's35a2_1': r'$\mathbf{t}^{T}\mathbf{y} \;=\; (\mathbf{X}\mathbf{w})^{T}\mathbf{y} \;=\; \mathbf{w}^{T}(\mathbf{X}^{T}\mathbf{y}) \;=\; 3\,w_1+4\,w_2$',
's35a2_2': [r'$\mathbf{w}=(1,0):\ \ 3\,,\qquad \mathbf{w}=(0,1):\ \ 4\,,\qquad \mathbf{w}=(0.6,\,0.8):\ \ 1.8+3.2 = 5$'],
's35a2_3': r'$\mathbf{t}^{T}\mathbf{y} \;=\; 3\cos\theta+4\sin\theta \;=\; 5\cos(\theta-53.13^{\circ})$',
's35a2_4': r'$\dfrac{(3,\ 4)}{5} \;=\; (\cos 53.13^{\circ},\ \sin 53.13^{\circ})\,,\qquad \cos(\theta-53.13^{\circ})\leq 1$',
# ---- 35A.3
's35a3_1': r'$\mathtt{coef\_}_j \;=\; \dfrac{s_y}{s_j}\,B_j\,:\qquad \dfrac{8}{2}\cdot 0.5 = 2\,,\qquad \dfrac{8}{4}\cdot 0.25 = 0.5$',
's35a3_2': r'$\hat{y} \;=\; 20 + (12-10)\cdot 2 + (9-5)\cdot 0.5 \;=\; 20+4+2 \;=\; 26$',
's35a3_3': r'$20 + 8\left[\dfrac{12-10}{2}\cdot 0.5+\dfrac{9-5}{4}\cdot 0.25\right] \;=\; 20+8\,(0.5+0.25) \;=\; 26$',
# ---- 35C.1
's35c1_1': r'$\dfrac{X_{j}^{T}\mathbf{y}}{n-1} \;=\; s_j\; s_y\; r_j\,,\qquad j=1,2$',
's35c1_2': r'$\mathbf{w}_1 \;\propto\; \left(s_1 r_1,\ \ s_2 r_2\right)\;=\;(34.8\times 0.018,\ \ 0.92\times 0.964) \;\propto\; (0.61,\ 0.88)$',
# ---- questions
's35q1_1': r'$\mathrm{cov}(\mathbf{t},\mathbf{y}) \;=\; \mathrm{sd}(\mathbf{t})\;\mathrm{sd}(\mathbf{y})\;\mathrm{corr}(\mathbf{t},\mathbf{y})$',
}
E.pop('s35a1_8')
mkeq(E, '.')
print('solution equations ok:', len(E))
