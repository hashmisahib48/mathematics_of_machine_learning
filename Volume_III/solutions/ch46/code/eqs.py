# Equation images of the Chapter 46 solutions. Run: python3 code/eqs.py  (writes sol/ch46/s46*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq, matimg
E = {
# ---- 46A.1
's46a1_1': r'$\det\mathbf{S} \;=\; 4\times3-2\times2 \;=\; 8$',
's46a1_2': r'$\Vert\mathbf{A}\Vert^{2} \;=\; 2^2+(-1)^2 \;=\; 5\,,\qquad \Vert\mathbf{B}\Vert^{2} \;=\; 2^2+1^2 \;=\; 5$',
's46a1_3a': r'$d^{2}(\mathbf{A}) \;=\; \dfrac{3\cdot 2^{2}-4\cdot 2\cdot(-1)+4\cdot(-1)^{2}}{8} \;=\; \dfrac{12+8+4}{8} \;=\; 3$',
's46a1_3b': r'$d^{2}(\mathbf{B}) \;=\; \dfrac{3\cdot 2^{2}-4\cdot 2\cdot 1+4\cdot 1^{2}}{8} \;=\; \dfrac{12-8+4}{8} \;=\; 1$',
's46a1_4': r'$P(\chi^{2}_{2}\geq 3) \;=\; e^{-3/2} \;=\; 0.223\,,\qquad P(\chi^{2}_{2}\geq 1) \;=\; e^{-1/2} \;=\; 0.607$',
# ---- 46A.2
's46a2_1': [r'$\det(\mathbf{S}-\lambda\mathbf{I}) \;=\; (5-\lambda)^{2}-16 \;=\; 0 \quad\Longrightarrow\quad \lambda \;=\; 5\pm4 \;=\; 9\,,\ 1$',
            r'$\kappa \;=\; 9/1 \;=\; 9\,,\qquad \hat{\mu} \;=\; \mathrm{tr}\,\mathbf{S}/p \;=\; (5+5)/2 \;=\; 5$'],
's46a2_3': [r'$\lambda_{1}^{\prime} \;=\; 9+0.25\,(5-9) \;=\; 8\,,\qquad \lambda_{2}^{\prime} \;=\; 1+0.25\,(5-1) \;=\; 2$',
            r'$\kappa^{\prime} \;=\; 8/2 \;=\; 4$'],
# ---- 46A.3
's46a3_1a': r'$\rho_{12\cdot3} \;=\; \dfrac{-\Theta_{12}}{\sqrt{\Theta_{11}\Theta_{22}}} \;=\; \dfrac{1}{\sqrt{2\cdot2}} \;=\; 0.5$',
's46a3_1b': r'$\rho_{13\cdot2} \;=\; \dfrac{-\Theta_{13}}{\sqrt{\Theta_{11}\Theta_{33}}} \;=\; \dfrac{0}{2} \;=\; 0$',
's46a3_2': [r'$\mathbf{\Theta}\,\boldsymbol{\sigma}_1=\mathbf{e}_1:\qquad 2\sigma_{11}-\sigma_{21}=1\,,\quad -\sigma_{11}+2\sigma_{21}-\sigma_{31}=0\,,\quad -\sigma_{21}+2\sigma_{31}=0$',
            r'$\sigma_{21}=2\sigma_{31}\,,\quad \sigma_{11}=2\sigma_{21}-\sigma_{31}=3\sigma_{31}\,,\quad 6\sigma_{31}-2\sigma_{31}=1 \;\Longrightarrow\; \boldsymbol{\sigma}_1=(\frac{3}{4},\ \frac{1}{2},\ \frac{1}{4})^{T}$'],
's46a3_4a': r'$r_{12} \;=\; \dfrac{0.5}{\sqrt{0.75\times1}} \;=\; 0.577\,,\qquad r_{23}=0.577$',
's46a3_4b': r'$r_{13} \;=\; \dfrac{0.25}{\sqrt{0.75\times0.75}} \;=\; \dfrac{1}{3} \;=\; 0.333 \;=\; r_{12}\,r_{23}$',
# ---- 46A.4
's46a4_1': r'$\alpha_{\max} \;=\; \max_{i\neq j}\,\vert S_{ij}\vert \;=\; \max\,(0.6,\ 0.2,\ 0.5) \;=\; 0.6$',
's46a4_2': r'$\mathbf{\Theta} \;=\; \mathrm{diag}\left(\dfrac{1}{S_{11}},\ \dfrac{1}{S_{22}},\ \dfrac{1}{S_{33}}\right) \;=\; \mathbf{I}\,,\qquad \vert W_{ij}-S_{ij}\vert \;=\; \vert S_{ij}\vert \;\leq\; \alpha$',
# ---- 46C.2
's46c2_1': r'$\dfrac{1+d_i^{2}}{n} \;=\; h_{ii} \;\leq\; 1 \quad\Longrightarrow\quad d_i^{2} \;\leq\; n-1$',
}
mkeq(E, '.')
matimg('s46a1_5', [r'$\mathbf{\Theta}\;=\;\mathbf{S}^{-1}\;=\;\dfrac{1}{8}$', [['3', '-2'], ['-2', '4']]], '.')
matimg('s46a2_2', [r'$0.75\,\mathbf{S}+1.25\,\mathbf{I}\;=\;$', [['3.75+1.25', '3'], ['3', '3.75+1.25']], r'$=$', [['5', '3'], ['3', '5']]], '.')
matimg('s46a3_3', [r'$\mathbf{\Sigma}\;=\;\mathbf{\Theta}^{-1}\;=\;\dfrac{1}{4}$', [['3', '2', '1'], ['2', '4', '2'], ['1', '2', '3']], r'$=$', [['0.75', '0.5', '0.25'], ['0.5', '1', '0.5'], ['0.25', '0.5', '0.75']]], '.')
print('solution equations ok:', len(E) + 3)
