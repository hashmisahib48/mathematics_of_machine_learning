# Equation images of the Chapter 2 solutions. Run: python3 code/eqs.py  (writes sol/ch02/s2*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq, matimg
E = {
# ---- 2A.1
's2a1_1': r'$\Vert\mathbf{x}\Vert_2 \;=\; \sqrt{2^2+(-3)^2+6^2} \;=\; \sqrt{4+9+36} \;=\; \sqrt{49} \;=\; 7$',
's2a1_2': [r'$\Vert\mathbf{x}\Vert_1 \;=\; |2|+|-3|+|6| \;=\; 2+3+6 \;=\; 11$', r'$\Vert\mathbf{x}\Vert_\infty \;=\; \max(2,\,3,\,6) \;=\; 6$'],
's2a1_3': r'$\mathbf{u} \;=\; \dfrac{\mathbf{x}}{7} \;=\; \left(\dfrac{2}{7},\,-\dfrac{3}{7},\,\dfrac{6}{7}\right) \;=\; (0.286,\,-0.429,\,0.857)$',
's2a1_4': [r'$\mathbf{x}-\mathbf{z} \;=\; (2-1,\;-3+3,\;6-2) \;=\; (1,\,0,\,4)$', r'$\Vert\mathbf{x}-\mathbf{z}\Vert_2 \;=\; \sqrt{1+0+16} \;=\; \sqrt{17} \;=\; 4.123$'],
# ---- 2A.2
's2a2_1': r'$c \;=\; \dfrac{\mathbf{a}^{T}\mathbf{b}}{\mathbf{a}^{T}\mathbf{a}} \;=\; \dfrac{1\times4+2\times3}{1^2+2^2} \;=\; \dfrac{10}{5} \;=\; 2$',
's2a2_2': [r'$\mathbf{p} \;=\; c\,\mathbf{a} \;=\; 2\,(1,\,2) \;=\; (2,\,4)$', r'$\mathbf{r} \;=\; \mathbf{b}-\mathbf{p} \;=\; (4-2,\;3-4) \;=\; (2,\,-1)$'],
's2a2_3': [r'$\mathbf{a}^{T}\mathbf{r} \;=\; 1\times2+2\times(-1) \;=\; 0$', r'$\Vert\mathbf{b}\Vert^2 = 25 \;=\; 20+5 \;=\; \Vert\mathbf{p}\Vert^2+\Vert\mathbf{r}\Vert^2$'],
# ---- 2A.3
's2a3_1': r'$\det\mathbf{M} \;=\; ad-bc \;=\; 3\times2-1\times5 \;=\; 6-5 \;=\; 1$',
# ---- 2A.4
's2a4_1': r'$\lambda^2-6\lambda+8 \;=\; 0 \quad\Longrightarrow\quad (\lambda-4)(\lambda-2)=0\,,\qquad \lambda_1=4\,,\;\lambda_2=2$',
's2a4_2': [r'$\mathrm{tr}\,\mathbf{S} = 3+3 = 6 = \lambda_1+\lambda_2$', r'$\det\mathbf{S} = 3\times3-1\times1 = 8 = \lambda_1\lambda_2$'],
# ---- questions: none
}
mkeq(E, '.')
matimg('s2a3_2', [r'$\mathbf{M}^{-1} \;=\; \dfrac{1}{1}$', [['2', '-1'], ['-5', '3']], r'$=$', [['2', '-1'], ['-5', '3']]], '.')
matimg('s2a3_3', [r'$\mathbf{x} \;=\; \mathbf{M}^{-1}\mathbf{b} \;=\;$', [['2', '-1'], ['-5', '3']], [['1'], ['2']], r'$=$', [['2-2'], ['-5+6']], r'$=$', [['0'], ['1']]], '.')
matimg('s2a3_4', [r'$\mathbf{M}\mathbf{x} \;=\;$', [['3', '1'], ['5', '2']], [['0'], ['1']], r'$=$', [['1'], ['2']], r'$=\;\mathbf{b}$'], '.')
matimg('s2a4_3', [r'$\mathbf{S}-4\mathbf{I} \;=\;$', [['-1', '1'], ['1', '-1']], r'$\Rightarrow\;\; v_1=v_2\,,\quad \mathbf{q}_1=\dfrac{(1,\,1)}{\sqrt{2}}$'], '.')
matimg('s2a4_4', [r'$\mathbf{S}-2\mathbf{I} \;=\;$', [['1', '1'], ['1', '1']], r'$\Rightarrow\;\; v_2=-v_1\,,\quad \mathbf{q}_2=\dfrac{(1,\,-1)}{\sqrt{2}}$'], '.')
matimg('s2a4_5', [r'$\mathbf{S} \;=\; 4\cdot\dfrac{1}{2}$', [['1', '1'], ['1', '1']], r'$+\;2\cdot\dfrac{1}{2}$', [['1', '-1'], ['-1', '1']], r'$=$', [['2', '2'], ['2', '2']], r'$+$', [['1', '-1'], ['-1', '1']]], '.')
print('solution equations ok')
