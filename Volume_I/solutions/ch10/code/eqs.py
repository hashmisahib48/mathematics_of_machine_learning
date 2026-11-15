# Equation images of the Chapter 10 solutions. Run: python3 code/eqs.py  (writes sol/ch10/s10*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq, matimg
from PIL import Image
E = {
# ---- questions
's10q2_1': [r'$J_{\mathrm{s}}(b,w) \;=\; \sum_i (60\,y_i-b-w\,x_i)^{2}+\alpha\,(b^{2}+w^{2})$', r'$=\; 3600\,\left[\sum_i (y_i-b/60-(w/60)\,x_i)^{2}+\alpha\,((b/60)^{2}+(w/60)^{2})\right]$'],
's10q3_1': [r'$\left(2\,\mathbf{X}_c^{T}\mathbf{X}_c+\alpha\,\mathbf{I}\right)\mathbf{w} \;=\; 2\,\mathbf{X}_c^{T}\mathbf{y}_c$', r'$\Longleftrightarrow\quad \left(\mathbf{X}_c^{T}\mathbf{X}_c+\dfrac{\alpha}{2}\,\mathbf{I}\right)\mathbf{w} \;=\; \mathbf{X}_c^{T}\mathbf{y}_c$'],
's10q4_1': r'$c_{+}=\dfrac{w_1+w_2}{\sqrt{2}}\,\ \ (\lambda_{+}\approx 100)\,,\qquad c_{-}=\dfrac{w_1-w_2}{\sqrt{2}}\,\ \ (\lambda_{-}\approx 0.05)$',
's10q4_2': r'$\mathrm{Var}(c_j) \;=\; \dfrac{\sigma^{2}}{\lambda_j}\ \ \mathrm{(least\ squares)}\,,\qquad \dfrac{\sigma^{2}\lambda_j}{(\lambda_j+\alpha)^{2}}\ \ \mathrm{(ridge)}$',
# ---- 10A.1
's10a1_1': r'$w \;=\; \dfrac{S_{xy}}{S_{xx}+\alpha} \;=\; \dfrac{8}{5+\alpha}\,,\qquad b \;=\; \bar{y}-w\,\bar{x} \;=\; 3.5-1.5\,w$',
's10a1_2': r'$\dfrac{8}{5+\alpha} \;=\; 1 \quad\Longrightarrow\quad \alpha \;=\; 8-5 \;=\; 3\,,\qquad b \;=\; 3.5-1.5 \;=\; 2.0$',
# ---- 10A.2
's10a2_1': ('mat', [r'$\mathbf{X}_c^{T}\mathbf{X}_c \;=$', [['5', '4'], ['4', '5']], r'$,\qquad \mathbf{X}_c^{T}\mathbf{y}_c \;=$', [['9'], ['3']]]),
's10a2_2': ('mat', [r'$\hat{\mathbf{w}} \;=\; \dfrac{1}{9}$', [['5', '-4'], ['-4', '5']], [['9'], ['3']], r'$=\; \dfrac{1}{9}$', [['33'], ['-21']], r'$=$', [['3.667'], ['-2.333']]]),
's10a2_3': ('mat', [r'$\hat{\mathbf{w}} \;=\; \dfrac{1}{20}$', [['6', '-4'], ['-4', '6']], [['9'], ['3']], r'$=\; \dfrac{1}{20}$', [['42'], ['-18']], r'$=$', [['2.1'], ['-0.9']]]),
's10a2_5': r'$\lambda \;=\; 5\pm 4 \;=\; 9\ \mathrm{and}\ 1\,,\qquad \mathbf{v}_1=\dfrac{(1,\,1)}{\sqrt{2}}\,,\quad \mathbf{v}_2=\dfrac{(1,\,-1)}{\sqrt{2}}$',
's10a2_6': r'$f_1 \;=\; \dfrac{9}{9+1} \;=\; 0.9\,,\qquad f_2 \;=\; \dfrac{1}{1+1} \;=\; 0.5\,,\qquad \mathrm{df} \;=\; 0.9+0.5 \;=\; 1.4$',
's10a2_7': r'$\mathbf{v}_1^{T}\mathbf{X}_c^{T}\mathbf{y}_c \;=\; \dfrac{9+3}{\sqrt{2}} \;=\; 8.485\,,\qquad \mathbf{v}_2^{T}\mathbf{X}_c^{T}\mathbf{y}_c \;=\; \dfrac{9-3}{\sqrt{2}} \;=\; 4.243$',
's10a2_8': r'$c_1 \;=\; \dfrac{8.485}{9+1} \;=\; 0.8485\,,\qquad c_2 \;=\; \dfrac{4.243}{1+1} \;=\; 2.1213$',
's10a2_9': r'$\hat{\mathbf{w}} \;=\; c_1\mathbf{v}_1+c_2\mathbf{v}_2 \;=\; (0.6+1.5,\ \ 0.6-1.5) \;=\; (2.1,\,-0.9)$',
# ---- 10A.3
's10a3_1': r'$\hat{\mathbf{w}} \;=\; \left(\mathbf{I}+\alpha\,\mathbf{I}\right)^{-1}\mathbf{X}_c^{T}\mathbf{y}_c \;=\; \dfrac{1}{1+\alpha}\,\hat{\mathbf{w}}_0\,,\qquad \hat{\mathbf{w}}_0 \;=\; \mathbf{X}_c^{T}\mathbf{y}_c$',
's10a3_2': [r'$\alpha=1:\ \ \dfrac{(3,\,-2,\,0.5)}{2} \;=\; (1.5,\,-1,\,0.25)$', r'$\alpha=3:\ \ \dfrac{(3,\,-2,\,0.5)}{4} \;=\; (0.75,\,-0.5,\,0.125)$'],
's10a3_3': r'$\mathrm{df} \;=\; \sum_{j=1}^{3}\dfrac{1}{1+\alpha} \;=\; \dfrac{3}{1+\alpha} \;=\; 1.5\ \ (\alpha=1)\,,\quad 0.75\ \ (\alpha=3)$',
# ---- 10A.4
's10a4_1': r'$S_{xx}+\alpha \;=\; 5+5 \;=\; 10\,,\qquad w \;=\; \dfrac{8}{10} \;=\; 0.8\,,\qquad h_{ii} \;=\; \dfrac{1}{4}+\dfrac{(x_i-1.5)^{2}}{10}$',
's10a4_2': r'$\mathrm{LOO} \;=\; \dfrac{1}{4}\,(6.131+0.019+0.019+6.131) \;=\; \dfrac{12.301}{4} \;=\; 3.075$',
's10a4_3': [r'$\bar{x}=2\,,\quad \bar{y}=\dfrac{3+4+6}{3}=4.3333\,,\quad S_{xx}=2\,,\quad S_{xy}=3\,,\quad w=\dfrac{3}{2+5}=0.4286$',
            r'$\hat{y}(0) \;=\; 4.3333+0.4286\,(0-2) \;=\; 3.4762\,,\qquad e_1 \;=\; 1-3.4762 \;=\; -2.4762$'],
# ---- 10B.1
's10b1_1': r'$\hat{\mathbf{w}}(\alpha) \;=\; \sum_{j}\dfrac{\sigma_j}{\sigma_j^{2}+\alpha}\,\left(\mathbf{u}_j^{T}\mathbf{y}_c\right)\mathbf{v}_j$',
# ---- 10C.1
's10c1_1': [r'$w \;=\; \dfrac{S_{xy}}{S_{xx}+\alpha} \;=\; \dfrac{17}{10+10} \;=\; 0.85$', r'$b \;=\; \bar{y}-\bar{x}\,w \;=\; 106-3\times 0.85 \;=\; 103.45$'],
's10c1_2': ('mat', [r'$\left(\mathbf{X}_1^{T}\mathbf{X}_1+\alpha\,\mathbf{I}\right)$', [['15', '15'], ['15', '65']], r'$\begin{pmatrix}b\\ w\end{pmatrix}$'][:2] + [[['b'], ['w']], r'$=$', [['530'], ['1607']]]),
# ---- 10C.3
's10c3_1': r'$\hat{\mathbf{w}} \;=\; \mathbf{X}_c^{T}\left(\mathbf{X}_c\mathbf{X}_c^{T}+\alpha\,\mathbf{I}\right)^{-1}\mathbf{y}_c \;\longrightarrow\; \mathbf{X}_c^{+}\,\mathbf{y}_c \qquad (\alpha\to 0)$',
's10c3_2': r'$\hat{\mathbf{w}} \;\approx\; \dfrac{1}{\alpha}\,\mathbf{X}_c^{T}\mathbf{y}_c \;\longrightarrow\; \mathbf{0} \qquad (\alpha\to\infty)$',
's10c3_3': r'$\mathrm{df}(\alpha) \;=\; \sum_{j=1}^{29}\dfrac{\sigma_j^{2}}{\sigma_j^{2}+\alpha}\ \ \longrightarrow\ \ 29 \quad (\alpha\to 0)$',
}
for k, v in E.items():
    if isinstance(v, tuple): matimg(k, v[1], '.')
    else: mkeq({k: v}, '.')
for k in sorted(E):
    w = Image.open(f'{k}.png').size[0] / 300
    if w > 5.2: print('TOO WIDE', k, round(w, 2))
print(len(E), 'equation images written')
