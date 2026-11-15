# Equation images of the Chapter 38 solutions. Run: python3 code/eqs.py  (writes sol/ch38/s38*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 38A.1
's38a1_1': r'$\sum_{i}\left|N_{\varepsilon}(i)\right| \;=\; 2+3+3+3+3+2+1 \;=\; 17 \;=\; 7+2\times5$',
's38a1_2': r'$\left|N_{\varepsilon}(i)\right|\;\geq\;m=3\quad\Longleftrightarrow\quad c_{3}(i)\;\leq\;\varepsilon=1.5$',
# ---- 38A.2
's38a2_1': [r'$c_{3}(a)=1.5\,,\qquad$ distances from $a$: $0,\ 1,\ 1.5,\ 4$',
            r'$c_{3}(b)=1.0\,,\qquad$ distances from $b$: $0,\ 0.5,\ 1,\ 3$',
            r'$c_{3}(c)=1.5\,,\qquad$ distances from $c$: $0,\ 0.5,\ 1.5,\ 2.5$',
            r'$c_{3}(d)=3.0\,,\qquad$ distances from $d$: $0,\ 2.5,\ 3,\ 4$'],
's38a2_2': r'$d_{\mathrm{mr}}(b,d)\;=\;\max\{\,d(b,d),\ c_{3}(b),\ c_{3}(d)\,\}\;=\;\max\{3,\ 1,\ 3\}\;=\;3$',
's38a2_3': [r'$\mathrm{MST}\ \mathrm{of}\ d_{\mathrm{mr}}:\quad 1.5+1.5+3 \;=\; 6.0$',
            r'$\mathrm{MST}\ \mathrm{of}\ d:\quad 0.5+1+2.5 \;=\; 4.0$'],
# ---- 38C.1
's38c1_1': r'$d(0.4,\,0.9)\;=\;0.5\;=\;\varepsilon\;=\;d(0.9,\,1.4)$',
# ---- 38C.2
's38c2_1': r'$S(\mathrm{root})\;=\;281.4\;\geq\;S(C_{1})+S(C_{2})\;=\;1.46+3.72\;=\;5.18$',
# ---- 38B.2
's38b2_1': r'$\mathrm{share}_{j}\;=\;\dfrac{s_{j}^{2}}{\sum_{k=1}^{13}s_{k}^{2}}\;,\qquad \mathrm{standardised:}\quad s_{j}=1\;\Rightarrow\;\mathrm{share}_{j}=\dfrac{1}{13}$',
}
mkeq(E, '.'); print('solution equations ok:', len(E))
