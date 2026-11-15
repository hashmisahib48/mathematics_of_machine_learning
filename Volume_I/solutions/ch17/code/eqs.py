# Equation images of the Chapter 17 solutions. Run: python3 code/eqs.py  (writes sol/ch17/s17*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
from PIL import Image
E = {
# ---- 17A.1
's17a1_1': [r'$z_3 \;=\; 1\times(3\cdot0+4\cdot0+0) \;=\; 0 \;\leq\; 0 \qquad\Longrightarrow\qquad \mathrm{update}$',
            r'$\mathbf{w}=(0,0)+1\cdot(3,4)=(3,4)\,,\qquad b=0+1=1$'],
's17a1_2': [r'$z_1 = 4+6-3 = 7\,,\qquad z_2 = -(1+0-3) = 2$',
            r'$z_3 = 3+8-3 = 8\,,\qquad z_4 = -(0+2-3) = 1$'],
's17a2_1': [r'$f_1 = 0.3\cdot4+0.3\cdot3-1.2 = 0.9\,,\quad f_2 = 0.3\cdot1+0-1.2 = -0.9$',
            r'$f_3 = 0.3\cdot3+0.3\cdot4-1.2 = 0.9\,,\quad f_4 = 0+0.3\cdot1-1.2 = -0.9$'],
's17a2_2': r'$J_{(a)} \;=\; \dfrac{0+0+0+0}{4} + \dfrac{0.1}{2}\,(2^2+1^2) \;=\; 0 + 0.25 \;=\; 0.25$',
's17a2_3': r'$J_{(b)} \;=\; \dfrac{4\times0.1}{4} + \dfrac{0.1}{2}\,(0.3^2+0.3^2) \;=\; 0.1 + 0.009 \;=\; 0.109$',
's17a3_1': [r'$z_1 \;=\; 1\times(0\cdot4+0\cdot3+0) \;=\; 0$',
            r'$L^{\prime}(0) \;=\; -\dfrac{1}{1+e^{0}} \;=\; -0.5$'],
's17a3_2': [r'$\mathbf{w} \;\leftarrow\; (0,0) - 0.5\times(-0.5)\times(+1)\times(4,3) \;=\; (1,\;0.75)$',
            r'$b \;\leftarrow\; 0 - 0.5\times(-0.5)\times(+1) \;=\; 0.25$'],
's17a3_3': [r'$z_2 \;=\; -1\times(1\cdot1+0.75\cdot0+0.25) \;=\; -1.25$',
            r'$L^{\prime}(-1.25) \;=\; -\dfrac{1}{1+e^{-1.25}} \;=\; -\dfrac{1}{1.2865} \;=\; -0.7773$'],
's17a3_4': [r'$\mathbf{w} \;\leftarrow\; (1,0.75) - 0.5\times(-0.7773)\times(-1)\times(1,0)$',
            r'$\;=\; (1-0.3886,\;0.75) \;=\; (0.6114,\;0.75)$',
            r'$b \;\leftarrow\; 0.25 - 0.5\times(-0.7773)\times(-1) \;=\; -0.1386$'],
's17a4_1': r'$\eta_1 \;=\; \alpha^{-1/4} \;=\; (10^{-4})^{-1/4} \;=\; 10$',
's17a4_2': r'$t_0 \;=\; \dfrac{1}{\alpha\,\eta_1} - 1 \;=\; \dfrac{1}{10^{-4}\times10} - 1 \;=\; 1000-1 \;=\; 999$',
's17a4_3': r'$\eta_{1000} \;=\; \dfrac{1}{10^{-4}\times(1000+999)} \;=\; \dfrac{1}{0.1999} \;=\; 5.0025$',
's17a4_4': r'$\eta_{100000} \;=\; \dfrac{1}{10^{-4}\times(100000+999)} \;=\; \dfrac{1}{10.0999} \;=\; 0.0990$',
's17a4_5': r'$\eta_1 \;=\; \dfrac{1}{10^{-4}\times(1+999)} \;=\; \dfrac{1}{0.1} \;=\; 10$',
# ---- 17B.1
's17b1_1': r'$\bar{\mathbf{w}}_t \;=\; \bar{\mathbf{w}}_{t-1} + \dfrac{1}{t}\left(\mathbf{w}_t - \bar{\mathbf{w}}_{t-1}\right)$',
# ---- 17C.1, 17C.3
's17c1_1': r'$\mathrm{no\ progress\ if}\quad S_k \;>\; S_{\mathrm{best}} - \mathrm{tol}\times n\,, \qquad S_k = \sum_{i}\max(0,\,-z_i)$',
's17c3_1': r'$\dfrac{d}{dz}\,(1-z)^2 \;=\; 2\,(z-1) \;=\; +4 \qquad\mathrm{at}\;\; z=3$',
# ---- questions
's17q2_1': [r'$J \;=\; \dfrac{1}{n}\sum_{i=1}^{n}\left[\,L(z_i) + \alpha R(\mathbf{w})\,\right]$',
            r'$\mathbf{w}\;\leftarrow\;(1-\eta\alpha)\,\mathbf{w} - \eta\,L^{\prime}(z_i)\,y_i\mathbf{x}_i \qquad (R = \frac{1}{2}\Vert\mathbf{w}\Vert^2)$'],
's17q2_2': r'$(1-\eta\alpha)^{n} \;\approx\; 1 - n\,\eta\alpha \qquad (0.99999^{426} = 0.99575)$',
's17q3_1': [r'$z^{\mathrm{new}} \;=\; z + \eta\,(\Vert\mathbf{x}\Vert^2+1)$',
            r'$\eta_1=10:\quad \Vert\mathbf{x}\Vert^2\approx30 \;\to\; +310\,, \qquad \Vert\mathbf{x}\Vert^2\approx10^{6} \;\to\; +10^{7}$'],
}
mkeq(E, '.'); print('solution equations ok:', len(E))
for k in E:
    with Image.open(k + '.png') as im:
        if im.size[0] > 1250: print('  too wide (will be shrunk):', k, im.size[0], 'px')
