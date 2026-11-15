# Equation images of the Chapter 1 solutions. Run: python3 code/eqs.py  (writes sol/ch01/s1*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 1A.1
's1a1_1': r'$\bar{x}_{\cdot 2} \;=\; \dfrac{5+4+1+6}{4} \;=\; \dfrac{16}{4} \;=\; 4$',
# ---- 1A.2
's1a2_1': r'$c^{*}_{\mathrm{sq}} \;=\; \bar{y} \;=\; \dfrac{1+2+3+4+5+6+28}{7} \;=\; \dfrac{49}{7} \;=\; 7$',
's1a2_2': [r'$J_{\mathrm{sq}} \;=\; \dfrac{(-6)^2+(-5)^2+(-4)^2+(-3)^2+(-2)^2+(-1)^2+21^2}{7}$', r'$=\; \dfrac{36+25+16+9+4+1+441}{7} \;=\; \dfrac{532}{7} \;=\; 76$'],
's1a2_3': r'$J_{\mathrm{abs}} \;=\; \dfrac{3+2+1+0+1+2+24}{7} \;=\; \dfrac{33}{7} \;\approx\; 4.714$',
# ---- 1A.3
's1a3_1': [r'$c_1 = 0-0.1\times(-10) = 1$', r'$c_2 = 1-0.1\times(-8) = 1.8$', r'$c_3 = 1.8-0.1\times(-6.4) = 2.44$'],
's1a3_2': [r'$c_{t}-\bar{y} \;=\; (1-2\eta)\,(c_{t-1}-\bar{y})$', r'$c_{t} \;=\; \bar{y}+(1-2\eta)^{t}\,(c_{0}-\bar{y}) \;=\; 5\,(1-0.8^{t})$'],
's1a3_3': r'$-1 \;<\; 1-2\eta \;<\; 1 \quad\Longleftrightarrow\quad 0 \;<\; \eta \;<\; 1$',
# ---- 1A.4
's1a4_1': [r'$\mathrm{SSE} \;=\; 0.25+0+0.25+0+2.25 \;=\; 2.75$', r'$\mathrm{MSE} \;=\; \dfrac{2.75}{5} \;=\; 0.55$', r'$\mathrm{MAE} \;=\; \dfrac{0.5+0+0.5+0+1.5}{5} \;=\; 0.5$'],
# ---- 1B.1
's1b1_1': r'$\dfrac{dJ_{\mathrm{abs}}}{dc} \;=\; \dfrac{N_{\mathrm{below}}-N_{\mathrm{above}}}{n} \;=\; \dfrac{3-3}{6} \;=\; 0 \qquad (4<c<5)$',
# ---- 1C.1
's1c1_1': r'$J_{\mathrm{LOO}} \;=\; \sqrt{\dfrac{1}{n}\sum_{i=1}^{n}\left(y_i-\hat{f}_{-i}(x_i)\right)^2}$',
}
mkeq(E, '.'); print('solution equations ok:', len(E))
