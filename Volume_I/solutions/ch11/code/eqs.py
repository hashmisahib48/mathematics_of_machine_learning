# Equation images of the Chapter 11 solutions. Run: python3 code/eqs.py  (writes sol/ch11/s11*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 11A.1
's11a1_1': r'$v \;=\; \dfrac{10}{5} \;=\; 2\,,\qquad c \;=\; \dfrac{17}{5} \;=\; 3.4$',
's11a1_2': r'$\alpha_{\max} \;=\; |c| \;=\; 3.4$',
's11a1_3': r'$w \;=\; \dfrac{S(c,\alpha)}{v} \;=\; \dfrac{3.4-2.5}{2} \;=\; \dfrac{0.9}{2} \;=\; 0.45$',
's11a1_4': r'$b \;=\; \bar{y}-w\,\bar{x} \;=\; 6-0.45\times 3 \;=\; 4.65$',
's11a1_5': r'$\hat{y}(6) \;=\; 4.65+0.45\times 6 \;=\; 7.35$',
's11a1_6': r'$\dfrac{1}{n}\sum_i (x_i-\bar{x})\,r_i \;=\; 2.5 \;=\; \alpha$',
# ---- 11A.2
's11a2_1': r'$c_2 \;=\; \dfrac{3+4+0+0+8}{5} \;=\; 3.0$',
's11a2_6': r'$\dfrac{1}{n}\sum_i x_{i1}x_{i2} \;=\; \dfrac{2+2+0+0+4}{5} \;=\; 1.6$',
's11a2_2': r'$z_1 \;=\; c_1 \;=\; 3.4\,,\qquad w_1 \;=\; \dfrac{S(3.4,\,2)}{2} \;=\; \dfrac{1.4}{2} \;=\; 0.7$',
's11a2_3': r'$z_2 \;=\; c_2-1.6\,w_1 \;=\; 3-1.12 \;=\; 1.88\,,\qquad |z_2|\leq 2 \;\Rightarrow\; w_2 \;=\; 0$',
's11a2_4': r'$z_1 \;=\; 2.0+2\times 0.7 \;=\; 3.4\,,\qquad w_1 \;=\; 0.7$',
's11a2_5': r'$b \;=\; 6-0.7\times 3-0\times 3 \;=\; 3.9$',
# ---- 11A.3
's11a3_1': r'$\lambda_1 \;=\; \alpha\rho \;=\; 0.5\,,\qquad \lambda_2 \;=\; \alpha(1-\rho) \;=\; 0.5$',
's11a3_2': r'$w \;=\; \dfrac{S(c,\lambda_1)}{v+\lambda_2} \;=\; \dfrac{3.4-0.5}{2+0.5} \;=\; \dfrac{2.9}{2.5} \;=\; 1.16$',
's11a3_3': r'$b \;=\; 6-1.16\times 3 \;=\; 2.52$',
's11a3_4': r'$c-(v+\lambda_2)\,w \;=\; 3.4-2.5\times 1.16 \;=\; 0.5 \;=\; \lambda_1$',
# ---- 11A.4
's11a4_1': r'$\Vert\mathbf{z}\Vert \;=\; \sqrt{6^2+8^2} \;=\; \sqrt{100} \;=\; 10$',
's11a4_2': r'$\mathbf{w} \;=\; \left(1-\dfrac{5}{10}\right)\dfrac{(6,\,8)}{2} \;=\; (1.5,\;2.0)$',
's11a4_3': r'$w_{1} \;=\; \dfrac{S(6,5)}{2} \;=\; \dfrac{1}{2} \;=\; 0.5\,,\qquad w_{2} \;=\; \dfrac{S(8,5)}{2} \;=\; \dfrac{3}{2} \;=\; 1.5$',
's11a4_4': r'$1-\dfrac{\alpha}{\Vert\mathbf{z}\Vert}\;\leq\; 0 \;\Longleftrightarrow\; \alpha \;\geq\; \Vert\mathbf{z}\Vert \;=\; 10$',
# ---- 11B
's11b1_1': r'$w_1 \;=\; \dfrac{c_1-\alpha}{v_1} \;=\; \dfrac{3400-1}{2\times 10^{6}} \;=\; 0.0016995$',
's11b1_2': r'$z_2 \;=\; 3-1600\times 0.0016995 \;=\; 0.281 \;<\; 1$',
's11b2_1': r'$\mathrm{score}_j \;=\; |\mathbf{x}_j^{\mathrm{T}}\mathbf{y}_c| \;=\; n\,|c_j|$',
's11b2_2': r'$w \;=\; \dfrac{\mathbf{x}_j^{\mathrm{T}}\mathbf{y}_c}{\mathbf{x}_j^{\mathrm{T}}\mathbf{x}_j} \;=\; \dfrac{150}{1000} \;=\; 0.15$',
# ---- 11C
's11c2_1': r'$\dfrac{1}{2n}\Vert\mathbf{y}_c-\mathbf{x}\,s\Vert^2+\alpha\,|s|\,,\qquad s=w_1+w_2\,,\quad w_1,\,w_2\geq 0$',
's11c2_2': r'$u \;=\; \dfrac{c-\lambda_1}{2v+\lambda_2} \;=\; \dfrac{3.4-0.5}{4+0.5} \;=\; \dfrac{2.9}{4.5} \;=\; 0.6444$',
's11c3_1': r'$\alpha_{\mathrm{start}} \;=\; \dfrac{\alpha_{\max}}{\rho}\,,\qquad \alpha_{\mathrm{end}} \;=\; \varepsilon\,\dfrac{\alpha_{\max}}{\rho}\,,\qquad \lambda_2^{\mathrm{end}} \;=\; \alpha_{\mathrm{end}}\,(1-\rho)$',
# ---- questions
}
mkeq(E, '.'); print('solution equations ok:', len(E))
