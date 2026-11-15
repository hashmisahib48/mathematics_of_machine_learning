# Equation images of the Chapter 42 solutions. Run: python3 code/eqs.py  (writes sol/ch42/s42*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq, matimg
E = {
# ---- 42A.1
's42a1_1': r'$\boldsymbol{\mu} \;=\; \left(\dfrac{6+3+5+6}{4}\,,\ \dfrac{2+2+3+5}{4}\right) \;=\; (5,\ 3)$',
's42a1_4': [r'$\det(\mathbf{S}-\lambda\mathbf{I}) \;=\; (2-\lambda)^2-1 \;=\; \lambda^2-4\lambda+3$', r'$=\; (\lambda-3)(\lambda-1) \;=\; 0$'],
's42a1_5': [r'$(\mathbf{S}-3\mathbf{I})\,\mathbf{w} \;=\; \mathbf{0}\ \Longrightarrow\ -w_1+w_2=0$', r'$\Longrightarrow\ \mathbf{w}_1 \;=\; \dfrac{1}{\sqrt{2}}\,(1,\ 1) \;=\; (0.7071,\ 0.7071)$'],
's42a1_6': r'$z_{i1} \;=\; \mathbf{w}_1^{T}\mathbf{x}_{ci} \;=\; \dfrac{x_{ci}+y_{ci}}{\sqrt{2}}$',
's42a1_7': [r'$\dfrac{\lambda_1}{\lambda_1+\lambda_2} \;=\; \dfrac{3}{3+1} \;=\; 0.75$', r'$\sum_i z_{i1}^2 \;=\; 4.5+4.5 \;=\; 9 \;=\; (n-1)\,\lambda_1 \;=\; 3\times3$'],
# ---- 42A.2
's42a2_1': r'$\sum_j\lambda_j \;=\; 4.2+2.4+1.6+1.1+0.7 \;=\; 10.0\,,\qquad r_j \;=\; \dfrac{\lambda_j}{10.0}$',
# ---- 42A.3
's42a3_1': r'$\lambda_j \;=\; \dfrac{\sigma_j^2}{n-1} \;=\; \dfrac{100}{100}\,,\ \dfrac{36}{100}\,,\ \dfrac{9}{100}\,,\ \dfrac{1}{100} \;=\; 1\,,\ 0.36\,,\ 0.09\,,\ 0.01$',
's42a3_2': r'$\sum_{j>2}\sigma_j^2 \;=\; 3^2+1^2 \;=\; 10 \;=\; (n-1)\sum_{j>2}\lambda_j \;=\; 100\times(0.09+0.01)$',
's42a3_3': r'$\dfrac{10}{101} \;=\; 0.099\,,\qquad \dfrac{\lambda_1+\lambda_2}{\sum_j\lambda_j} \;=\; \dfrac{1+0.36}{1.46} \;=\; 0.9315$',
# ---- 42A.4
's42a4_1': [r'$\mathbf{x}_c \;=\; (28-30,\ 6-6) \;=\; (-2,\ 0)$', r'$\mathbf{w}_1=\dfrac{(2,\,1)}{\sqrt{5}}\,,\qquad \mathbf{w}_2=\dfrac{(-1,\,2)}{\sqrt{5}}$'],
's42a4_2': r'$z_1 \;=\; \dfrac{(-2)(2)+0\cdot1}{\sqrt{5}} \;=\; -1.7889$',
's42a4_2b': r'$z_2 \;=\; \dfrac{(-2)(-1)+0\cdot2}{\sqrt{5}} \;=\; 0.8944$',
's42a4_3': [r'$\hat{\mathbf{x}} \;=\; (30,\ 6)+(-1.7889)\,(0.8944,\ 0.4472)$', r'$=\; (30-1.6,\ 6-0.8) \;=\; (28.4,\ 5.2)$'],
's42a4_4': r'$(28-28.4)^2+(6-5.2)^2 \;=\; 0.16+0.64 \;=\; 0.8 \;=\; z_2^2$',
# ---- questions
's42q1_1': r'$\Vert\mathbf{x}_{ci}\Vert^2 \;=\; (\mathbf{w}^{T}\mathbf{x}_{ci})^2 \;+\; \Vert\mathbf{x}_{ci}-(\mathbf{w}^{T}\mathbf{x}_{ci})\,\mathbf{w}\Vert^2$',
's42q1_2': [r'$\mathrm{weight\ on\ length:}\quad S_{xy}/S_{xx} \;=\; 2/5 \;=\; 0.4$', r'$\mathrm{length\ on\ weight:}\quad S_{yy}/S_{xy} \;=\; 2/2 \;=\; 1$'],
# ---- 42B.1
}
mkeq(E, '.')
matimg('s42a1_3', [r'$\mathbf{S}\;=\;\dfrac{1}{3}$', [['6', '3'], ['3', '6']], r'$\;=\;$', [['2', '1'], ['1', '2']]], '.')
print('solution equations ok:', len(E) + 1)
