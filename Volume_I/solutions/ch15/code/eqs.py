# Equation images of the Chapter 15 solutions. Run: python3 code/eqs.py  (writes sol/ch15/s15*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 15A.1
's15a1_1': r'$\varphi \;=\; (1,\ 3,\ -1,\ 3^2,\ 3\cdot(-1),\ (-1)^2) \;=\; (1,\ 3,\ -1,\ 9,\ -3,\ 1)$',
's15a1_2': [r'$\hat{y} \;=\; 1\cdot1+2\cdot3+(-1)(-1)+0.5\cdot9+1\cdot(-3)+2\cdot1$', r'$\;=\; 1+6+1+4.5-3+2 \;=\; 11.5$'],
's15a1_3': r'$m \;=\; \binom{p+d}{d} \;=\; \binom{3+3}{3} \;=\; \dfrac{6\cdot5\cdot4}{3\cdot2\cdot1} \;=\; 20$',
's15a1_4': r'$1+3+6+10 \;=\; 20\qquad(\mathrm{degree}\ 0,\ 1,\ 2,\ 3)$',
# ---- 15A.2
's15a2_1': [r'$\bar{y} \;=\; \dfrac{6+2+1+4+12}{5} \;=\; 5$', r'$\sum u_i^2 \;=\; 4+1+0+1+4 \;=\; 10\,,\qquad \sum q_i^2 \;=\; 4+1+4+1+4 \;=\; 14$'],
's15a2_2': r'$c_1 \;=\; \dfrac{\sum u_iy_i}{\sum u_i^2} \;=\; \dfrac{-12-2+0+4+24}{10} \;=\; \dfrac{14}{10} \;=\; 1.4$',
's15a2_3': r'$c_2 \;=\; \dfrac{\sum q_iy_i}{\sum q_i^2} \;=\; \dfrac{12-2-2-4+24}{14} \;=\; \dfrac{28}{14} \;=\; 2$',
's15a2_4': [r'$\hat{y} \;=\; 5+1.4\,(x-3)+2\,\left[(x-3)^2-2\right]$',
            r'$\;=\; (5-4.2+18-4) \;+\; (1.4-12)\,x \;+\; 2\,x^2 \;=\; 14.8-10.6\,x+2\,x^2$'],
's15a2_5': r'$\mathrm{SSE} \;=\; (-0.2)^2+0.4^2+0^2+(-0.4)^2+0.2^2 \;=\; 0.4$',
's15a2_6': r'$\hat{y}(6) \;=\; 14.8-10.6\cdot6+2\cdot36 \;=\; 14.8-63.6+72 \;=\; 23.2$',
# ---- 15A.3
's15a3_1': r'$B_{0,3}(x) \;=\; \dfrac{x-t_0}{t_3-t_0}\,B_{0,2}(x) \;+\; \dfrac{t_4-x}{t_4-t_1}\,B_{1,2}(x)\,,\qquad (t_0,\dots,t_4)=(0,1,2,3,4)$',
's15a3_2': r'$\dfrac{x-t_0}{t_3-t_0} \;=\; \dfrac{2-0}{3-0} \;=\; \dfrac{2}{3}\,,\qquad \dfrac{t_4-x}{t_4-t_1} \;=\; \dfrac{4-2}{4-1} \;=\; \dfrac{2}{3}$',
's15a3_3': r'$B(2) \;=\; \dfrac{2}{3}\cdot0.5+\dfrac{2}{3}\cdot0.5 \;=\; \dfrac{1}{3}+\dfrac{1}{3} \;=\; \dfrac{2}{3}$',
's15a3_4': [r'$B(1) \;=\; \dfrac{1-0}{3}\,B_{0,2}(1)+\dfrac{4-1}{3}\,B_{1,2}(1) \;=\; \dfrac{1}{3}\cdot0.5+1\cdot0 \;=\; \dfrac{1}{6}$', r'$B(3) \;=\; \dfrac{1}{6}\qquad\mathrm{(by\ symmetry\ about\ }x=2\mathrm{)}$'],
's15a3_5': r'$\dfrac{1}{6}+\dfrac{2}{3}+\dfrac{1}{6} \;=\; 1$',
# ---- 15A.4
's15a4_1': r'$\binom{6+4}{4} \;=\; \dfrac{10\cdot9\cdot8\cdot7}{4\cdot3\cdot2\cdot1} \;=\; 210$',
's15a4_2': r'$\binom{6}{0}+\binom{6}{1}+\binom{6}{2}+\binom{6}{3} \;=\; 1+6+15+20 \;=\; 42$',
's15a4_3': r'$p\,(n_{\mathrm{knots}}+d-1) \;-\; p \;=\; 4\,(7+3-1)-4 \;=\; 36-4 \;=\; 32$',
# ---- 15B.3, 15C.1, 15C.3
's15c1_1': r'$m \;=\; \binom{30+3}{3} \;=\; \dfrac{33\cdot32\cdot31}{6} \;=\; 5456\,,\qquad \dfrac{m-1}{n} \;=\; \dfrac{5455}{200} \;\approx\; 27.3$',
's15c1_2': r'$\hat{\mathbf{w}} \;=\; \mathbf{\Phi}_c^{+}\,\mathbf{y}_c\,,\qquad \operatorname{rank}\mathbf{\Phi}_c \;\leq\; n-1 \;=\; 199 \;<\; m-1$',
's15c3_1': r'$A\sin\dfrac{2\pi h}{24}+B\cos\dfrac{2\pi h}{24} \;=\; R\cos\dfrac{2\pi\,(h-h_0)}{24}\,,\qquad R=\sqrt{A^2+B^2}$',
# ---- questions
's15q1_1': r'$\hat{y} \;=\; b+w_1x+w_2x^2 \quad\mathrm{(linear\ in\ }w\mathrm{)}\qquad\mathrm{against}\qquad \hat{y} \;=\; a\,e^{c\,x} \quad\mathrm{(not)}$',
's15q2_1': r'$\dfrac{1-\bar{\phi}_0}{s_0} \;=\; \dfrac{1-1}{1} \;=\; 0\qquad(\bar{\phi}_0=1,\ s_0\ \mathrm{replaced\ by\ }1\ \mathrm{because\ the\ true\ one\ is\ }0)$',
's15q4_1': r'$\left(\mathbf{\Phi}^{T}\mathbf{\Phi}\right)_{jk} \;=\; \sum_i \varphi_j(x_i)\,\varphi_k(x_i) \;=\; 0 \qquad\mathrm{whenever}\ |j-k|>d$',
}
mkeq(E, '.'); print('solution equations ok:', len(E))
