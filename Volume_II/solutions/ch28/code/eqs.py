# Equation images of the Chapter 28 solutions. Run: python3 code/eqs.py  (writes sol/ch28/s28*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 28A.1
's28a1_1': [r'$t_{1}=\dfrac{x_{(2)}+x_{(3)}}{2}=\dfrac{5+6}{2}=5.5\,,\qquad t_{2}=\dfrac{x_{(4)}+x_{(5)}}{2}=\dfrac{8+9}{2}=8.5$',
             r'$t_{3}=\dfrac{x_{(6)}+x_{(7)}}{2}=\dfrac{12+14}{2}=13.0$'],
's28a1_2': r'$b(x)\;=\;\#\{k:\ t_{k}<x\}\,,\qquad b\in\{0,1,2,3\}$',
# ---- 28A.2
's28a2_1': r'$G=3-1-4+2=0\,,\qquad H=4\times2=8\,,\qquad \dfrac{G^{2}}{H+\lambda}=0$',
's28a2_2': r'$\mathrm{gain}_{0}=\dfrac{3^{2}}{2+1}+\dfrac{(-3)^{2}}{6+1}-0=3+1.2857=4.2857$',
's28a2_3': r'$\mathrm{gain}_{1}=\dfrac{2^{2}}{4+1}+\dfrac{(-2)^{2}}{4+1}-0=0.8+0.8=1.6000$',
's28a2_4': r'$\mathrm{gain}_{2}=\dfrac{(-2)^{2}}{6+1}+\dfrac{2^{2}}{2+1}-0=0.5714+1.3333=1.9048$',
's28a2_5': r'$w_{L}=-\dfrac{3}{2+1}=-1.0000\,,\qquad w_{R}=-\dfrac{-3}{6+1}=\dfrac{3}{7}=0.4286$',
# ---- 28A.3
's28a3_1': [r'$G_{b}^{\mathrm{large}}=G_{b}^{\mathrm{parent}}-G_{b}^{\mathrm{small}}$',
            r'$=(2.5-1.0,\ -1.5+0.5,\ 0.5-0.0,\ -0.5+0.5)$',
            r'$=(1.5,\ -1.0,\ 0.5,\ 0.0)$'],
's28a3_2': [r'$H_{b}^{\mathrm{large}}=(1.5-0.6,\ 1.2-0.4,\ 1.0-0.0,\ 1.3-0.5)$',
            r'$=(0.9,\ 0.8,\ 1.0,\ 0.8)$'],
's28a3_3': [r'$G=1.5-1.0+0.5+0.0=1.0$',r'$H=0.9+0.8+1.0+0.8=3.5$'],
's28a3_4': r'$w=-\dfrac{G}{H+\lambda}=-\dfrac{1.0}{3.5+1}=-0.2222$',
# ---- 28C.1
's28c1_1': r'$n_{\mathrm{leaf}}\;\geq\;m\ \ \Rightarrow\ \ n\;\geq\;2m\ \ \mathrm{for\ one\ split}\,,\qquad T\;\leq\;\lfloor n/m\rfloor$',
# ---- questions
's28q2_1': r'$g_{i}=\mathrm{sign}(F_{i}-y_{i})\,,\qquad h_{i}=1\,,\qquad G_{t}=n_{t}^{+}-n_{t}^{-}$',
}
mkeq(E, '.'); print('solution equations ok:', len(E))
