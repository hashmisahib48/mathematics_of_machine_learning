# Equation images of the Chapter 20 solutions. Run: python3 code/eqs.py  (writes sol/ch20/s20*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 20A.1
's20a1_1': [r'$\mathrm{score}_0 \;=\; \dfrac{1}{0.4} \;=\; 2.500$',
             r'$\mathrm{score}_1 \;=\; \dfrac{1}{0.6}+\dfrac{1}{2.4} \;=\; 1.667+0.417 \;=\; 2.083$'],
's20a1_2': r'$\hat{P}(0) \;=\; \dfrac{2.500}{2.500+2.083} \;=\; 0.545\,,\qquad \hat{P}(1) \;=\; \dfrac{2.083}{4.583} \;=\; 0.455$',
# ---- 20A.2
's20a2_1': [r'$d_{1}(\mathrm{A}) \;=\; |3|+|3| \;=\; 6\,,\qquad d_{1}(\mathrm{B}) \;=\; |4|+|0| \;=\; 4$'],
's20a2_2': [r'$d_{2}(\mathrm{A}) \;=\; \sqrt{3^2+3^2} \;=\; \sqrt{18} \;=\; 4.243\,,\qquad d_{2}(\mathrm{B}) \;=\; \sqrt{4^2+0^2} \;=\; 4$'],
's20a2_3': [r'$d_{\infty}(\mathrm{A}) \;=\; \max(3,\,3) \;=\; 3\,,\qquad d_{\infty}(\mathrm{B}) \;=\; \max(4,\,0) \;=\; 4$'],
's20a2_4': r'$2\cdot3^{q} \;=\; 4^{q} \quad\Longrightarrow\quad q \;=\; \dfrac{\ln 2}{\ln(4/3)} \;=\; 2.41$',
# ---- 20A.3
's20a3_1': [r'$k=2:\qquad \hat{y} \;=\; \dfrac{7+3}{2} \;=\; 5.000$',
            r'$\hat{y} \;=\; \dfrac{2\cdot 7+0.667\cdot 3}{2+0.667} \;=\; \dfrac{16.000}{2.667} \;=\; 6.000$'],
's20a3_2': [r'$k=3:\qquad \hat{y} \;=\; \dfrac{7+3+2}{3} \;=\; 4.000$',
            r'$\hat{y} \;=\; \dfrac{2\cdot 7+0.667\cdot 3+0.4\cdot 2}{2+0.667+0.4} \;=\; \dfrac{16.800}{3.067} \;=\; 5.478$'],
# ---- questions
's20q4_1': r'$\Vert\mathbf{u}-\mathbf{v}\Vert^{2} \;=\; 2-2\,\mathbf{u}^{T}\mathbf{v} \;=\; 2\,(1-\cos\theta) \qquad(\Vert\mathbf{u}\Vert=\Vert\mathbf{v}\Vert=1)$',
's20q5_1': r'$R^{*} \;\leq\; R_{1} \;\leq\; 2R^{*}(1-R^{*}) \quad\Longrightarrow\quad 0.050 \;\leq\; R_{1} \;\leq\; 2\times0.05\times0.95 \;=\; 0.095$',
}
mkeq(E, '.'); print('solution equations ok:', len(E))
