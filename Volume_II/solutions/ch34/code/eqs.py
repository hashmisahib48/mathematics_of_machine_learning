# Equation images of the Chapter 34 solutions. Run: python3 code/eqs.py  (writes sol/ch34/s34*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq, matimg
E = {
# ---- 34A.1
's34a1_1': r'$6\,f_u \;=\; 3\cdot 0 + 2\,f_v + 1\cdot 1 \;=\; 2f_v + 1$',
's34a1_2': r'$6\,f_v \;=\; 2\,f_u + 4\cdot 1 \;=\; 2f_u + 4$',
's34a1_3': r'$36\,f_v \;=\; 2\,(2f_v+1) + 24 \;\Longrightarrow\; 32\,f_v = 26 \;\Longrightarrow\; f_v = \dfrac{13}{16} = 0.8125$',
's34a1_4': r'$f_u \;=\; \dfrac{2\cdot 13/16 + 1}{6} \;=\; \dfrac{42/16}{6} \;=\; \dfrac{7}{16} = 0.4375$',
's34a1_5': r'$\dfrac{2\,f_u + 4}{6} \;=\; \dfrac{14/16 + 64/16}{6} \;=\; \dfrac{13}{16} \;=\; f_v$',
# ---- 34A.2
's34a2_1': r'$S_{12}=\dfrac{1}{\sqrt{1\cdot 4}}=0.5\,,\qquad S_{23}=\dfrac{3}{\sqrt{4\cdot 3}}=\dfrac{\sqrt{3}}{2}=0.8660$',
's34a2_2': [r'$\mathbf{F}_{2} \;=\; 0.5\,\left(S_{21}\mathbf{Y}_{1}+S_{23}\mathbf{Y}_{3}\right)+0.5\,\mathbf{Y}_{2}$',
            r'$=\; 0.5\,\left[(0.5,\,0)+(0,\,0.8660)\right]+(0,\,0) \;=\; (0.25,\ 0.4330)$'],
's34a2_3': r'$\mathbf{F}_{1} = 0.5\,S_{12}\mathbf{Y}_{2}+0.5\,\mathbf{Y}_{1} = (0.5,\ 0)\,,\qquad \mathbf{F}_{3} = 0.5\,S_{32}\mathbf{Y}_{2}+0.5\,\mathbf{Y}_{3} = (0,\ 0.5)$',
's34a2_4': [r'$\alpha S_{12}=0.25\,,\qquad \alpha S_{23}=0.4330$',
            r'$(1-\alpha)\mathbf{Y}_{1}=(0.5,\,0)\,,\qquad (1-\alpha)\mathbf{Y}_{3}=(0,\,0.5)$'],
's34a2_5': [r'$\mathrm{class\ 0:}\quad f_1-0.25f_2=0.5\,,\quad f_2-0.25f_1-0.4330f_3=0\,,$',
            r'$f_3-0.4330f_2=0$',
            r'$f_2\,(1-0.0625-0.1875) \;=\; 0.125 \;\Longrightarrow\; f_2 = \dfrac{1}{6} = 0.1667$'],
's34a2_6': [r'$\mathrm{class\ 1:}\quad f_1=0.25f_2\,,\quad f_3=0.5+0.4330f_2\,,$',
            r'$f_2\,(1-0.0625-0.1875)=0.4330\cdot 0.5$',
            r'$f_2 = \dfrac{0.2165}{0.75} = 0.2887$'],
's34a2_7': r'$\dfrac{(0.1667,\ 0.2887)}{0.1667+0.2887} \;=\; (0.366,\ 0.634)$',
# ---- 34A.3
's34a3_1': r'$f_2 \;=\; \dfrac{w_{21}f_1+w_{23}f_3}{d_2} \;=\; \dfrac{2\cdot 0+1\cdot 1}{3} \;=\; \dfrac{1}{3}$',
's34a3_3': r'$\det_{\,4,5} \;=\; 3\cdot 3-(-3)(-3) \;=\; 0\,,\qquad \mathbf{v}=(0,\,1,\,1)^{T}:\ \ (\mathbf{D}_{UU}-\mathbf{W}_{UU})\,\mathbf{v}=\mathbf{0}$',
# ---- 34A.4
's34a4_3': r'$f_2=\frac{1}{2} f_1+\frac{1}{2} f_2 \;\Rightarrow\; f_2=f_1=0\,,\qquad f_3=\frac{1}{2} f_2+\frac{1}{2} f_3 \;\Rightarrow\; f_3=f_2=0$',
# ---- 34C
's34c1_1': r'$d_i f_i \;=\; \sum_j w_{ij} f_j \;\leq\; \sum_j w_{ij}\, m \;=\; d_i\, m$',
's34c2_1': r'$h_i \;=\; \sum_j T_{ij}\,h_j \;\Longleftrightarrow\; d_i h_i = \sum_j w_{ij} h_j\,,\qquad h_1=0,\ h_5=1$',
's34c2_2': r'$h_3 \;=\; T_{32}h_2 + T_{34}h_4 \;=\; \dfrac{4}{5}\cdot\dfrac{2}{11}+\dfrac{1}{5}\cdot\dfrac{7}{11} \;=\; \dfrac{8+7}{55} \;=\; \dfrac{3}{11}$',
}
mkeq(E, '.')
matimg('s34a3_2', [r'$\mathbf{D}_{UU}-\mathbf{W}_{UU}=$', [['3', '0', '0'], ['0', '3', '-3'], ['0', '-3', '3']], r'$\ \ (\mathrm{nodes\ }2,4,5)$'], '.')
matimg('s34a4_1', [r'$\mathbf{W}=$', [['1', '1', '0', '0'], ['1', '1', '0', '0'], ['0', '1', '1', '0'], ['0', '0', '1', '1']]], '.')
matimg('s34a4_2', [r'$\mathbf{T}=$', [['\\frac{1}{2}', '\\frac{1}{2}', '0', '0'], ['\\frac{1}{2}', '\\frac{1}{2}', '0', '0'], ['0', '\\frac{1}{2}', '\\frac{1}{2}', '0'], ['0', '0', '\\frac{1}{2}', '\\frac{1}{2}']]], '.', rh=0.3)
print('solution equations ok:', len(E) + 3)
