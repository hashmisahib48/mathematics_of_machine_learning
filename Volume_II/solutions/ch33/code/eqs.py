# Equation images of the Chapter 33 solutions. Run: python3 code/eqs.py  (writes sol/ch33/s33*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 33A.1
's33a1_1': [r'$n \;=\; 400+300+200+100+50+50 \;=\; 1100$', r'$K\,n \;=\; 6\times 1100 \;=\; 6600$'],
's33a1_2': r'$\sum_{j<k}(n_j+n_k) \;=\; 2700+1600+800+300+100 \;=\; 5500$',
's33a1_3': r'$(K-1)\,n \;=\; 5\times 1100 \;=\; 5500$',
's33a1_4': [r'$\mathrm{OvR:}\quad K\,n^2 \;=\; 6\times 1100^2 \;=\; 7\,260\,000$',
            r'$\mathrm{OvO:}\quad \sum_{j<k}(n_j+n_k)^2 \;=\; 1\,505\,000+655\,000+215\,000$', r'$\qquad\qquad +\,45\,000+10\,000 \;=\; 2\,430\,000$'],
's33a1_5': r'$\dfrac{2\,430\,000}{7\,260\,000} \;=\; 0.3347 \;\approx\; \dfrac{1}{3}$',
# ---- 33A.2
's33a2_1': [r'$c_0 \;=\; -f_{01}-f_{02} \;=\; -0.8+0.6 \;=\; -0.2$', r'$c_1 \;=\; f_{01}-f_{12} \;=\; 0.8-0.3 \;=\; 0.5$', r'$c_2 \;=\; f_{02}+f_{12} \;=\; -0.6+0.3 \;=\; -0.3$'],
's33a2_2': [r'$F_0 \;=\; 1+\dfrac{-0.2}{3\,(0.2+1)} \;=\; 1-0.0556 \;=\; 0.9444$', r'$F_1 \;=\; 1+\dfrac{0.5}{3\,(0.5+1)} \;=\; 1+0.1111 \;=\; 1.1111$', r'$F_2 \;=\; 1+\dfrac{-0.3}{3\,(0.3+1)} \;=\; 1-0.0769 \;=\; 0.9231$'],
# ---- 33C.1, 33C.2
's33c1_1': r'$s_k(x) \;=\; \bar{t}_k+w_k\,(x-\bar{x})\,, \qquad \bar{t}_k=\dfrac{n_k}{n}\,, \qquad w_k=\dfrac{S_{xy,k}}{S_{xx}}$',
's33c1_2': r'$\max_k s_k(x) \;=\; \dfrac{1}{K}+\max_k\left[\,w_k\,(x-\bar{x})\,\right]$',
's33c2_1': r'$\left(\mathbf{X}^{T}\mathbf{X}+\alpha\mathbf{I}\right)\mathbf{u} \;=\; \mathbf{X}^{T}\mathbf{y}_1$',
's33c2_2': r'$\left(\mathbf{X}^{T}\mathbf{X}+\alpha\mathbf{I}\right)\mathbf{w}+c\,\mathbf{X}^{T}\mathbf{y}_1 \;=\; \mathbf{X}^{T}\mathbf{y}_2$',
's33c2_3': [r'$\left(\mathbf{X}^{T}\mathbf{X}+\alpha\mathbf{I}\right)(\mathbf{w}+c\,\mathbf{u}) \;=\; \mathbf{X}^{T}\mathbf{y}_2$', r'$\hat{y}_2 \;=\; \mathbf{x}^{T}\mathbf{w}+c\,\mathbf{x}^{T}\mathbf{u} \;=\; \mathbf{x}^{T}(\mathbf{w}+c\,\mathbf{u}) \;=\; \mathbf{x}^{T}\mathbf{v}$'],
# ---- questions
's33q1_1': r'$F_A-F_B \;>\; \left(v+1-\dfrac{1}{3}\right)-\left(v+\dfrac{1}{3}\right) \;=\; \dfrac{1}{3} \;>\; 0$',
}
mkeq(E, '.'); print('solution equations ok:', len(E))
