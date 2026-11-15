# Equation images of the Chapter 16 solutions. Run: python3 code/eqs.py  (writes sol/ch16/s16*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 16A.1
's16a1_1': r'$x=2:\quad z=-2+0.5\cdot2=-1\,,\qquad p=\dfrac{1}{1+e^{1}}=\dfrac{1}{3.7183}=0.2689$',
's16a1_1b': r'$x=4:\quad z=-2+0.5\cdot4=0\,,\qquad p=\dfrac{1}{1+e^{0}}=\dfrac{1}{2}=0.5000$',
's16a1_1c': r'$x=8:\quad z=-2+0.5\cdot8=2\,,\qquad p=\dfrac{1}{1+e^{-2}}=\dfrac{1}{1.1353}=0.8808$',
's16a1_2': r'$\mathrm{odds}(x=8) \;=\; e^{z} \;=\; e^{2} \;=\; 7.389\,,\qquad \dfrac{p}{1-p} \;=\; \dfrac{0.8808}{0.1192} \;=\; 7.389$',
's16a1_3': r'$\ln\dfrac{0.9}{0.1} \;=\; \ln 9 \;=\; 2.1972 \;=\; -2+0.5\,x$',
's16a1_3b': r'$x \;=\; \dfrac{2.1972+2}{0.5} \;=\; 8.394$',
# ---- 16A.2
's16a2_1': [r'$\ell_1=-\ln 0.9=0.1054\,,\qquad \ell_2=-\ln(1-0.2)=-\ln 0.8=0.2231$',
            r'$\ell_3=-\ln 0.6=0.5108\,,\qquad \ell_4=-\ln(1-0.5)=-\ln 0.5=0.6931$'],
's16a2_2': r'$J \;=\; 0.1054+0.2231+0.5108+0.6931 \;=\; 1.5325$',
's16a2_3': r'$J_{1/2} \;=\; 4\,\ln 2 \;=\; 4\times0.6931 \;=\; 2.7726$',
# ---- 16A.3
's16a3_1': r'$\mathbf{g} \;=\; \mathbf{X}^{T}(\mathbf{p}-\mathbf{y}) \;=\; (0,\ -7)$',
's16a3_2': [r'$(b,\,w) \;\leftarrow\; -\eta\,\mathbf{g} \;=\; -0.02\,(0,\,-7) \;=\; (0,\ 0.14)$',
            r'$(b,\,w) \;\leftarrow\; -0.05\,(0,\,-7) \;=\; (0,\ 0.35)$'],
's16a3_3': r'$\lambda \;=\; \dfrac{\mathrm{tr}\,\pm\sqrt{\mathrm{tr}^2-4\det}}{2} \;=\; \dfrac{53\pm\sqrt{53^2-4\cdot21}}{2} \;=\; \dfrac{53\pm52.2015}{2}$',
's16a3_3b': r'$\lambda_{\max} \;=\; 52.601\,,\quad \lambda_{\min} \;=\; 0.399\,,\qquad \eta \;<\; \dfrac{2}{\lambda_{\max}} \;=\; \dfrac{2}{52.601} \;=\; 0.0380$',
's16a3_4': r'$\det\mathbf{H} \;=\; 2\cdot51-9\cdot9 \;=\; 21\,,\qquad \mathrm{tr}\,\mathbf{H} \;=\; 2+51 \;=\; 53$',
's16a3_5': r'$J(\eta) \;\approx\; J_0-\eta\,\mathbf{g}^{T}\mathbf{g}+\dfrac{\eta^2}{2}\,\mathbf{g}^{T}\mathbf{H}\,\mathbf{g} \;=\; 5.5452-49\,\eta+1249.5\,\eta^2$',
's16a3_6': r'$\eta_{*} \;=\; \dfrac{\mathbf{g}^{T}\mathbf{g}}{\mathbf{g}^{T}\mathbf{H}\,\mathbf{g}} \;=\; \dfrac{49}{2499} \;=\; 0.0196$',
# ---- 16A.4
's16a4_1': r'$e^{0}=1\,,\qquad e^{1}=2.7183\,,\qquad e^{3}=20.0855\,,\qquad \sum_j e^{f_j} \;=\; 23.8038$',
's16a4_2': r'$\mathbf{P} \;=\; \left(\dfrac{1}{23.8038},\ \dfrac{2.7183}{23.8038},\ \dfrac{20.0855}{23.8038}\right) \;=\; (0.0420,\ 0.1142,\ 0.8438)$',
's16a4_3': r'$\ell \;=\; -\ln P_2 \;=\; -\ln 0.1142 \;=\; 2.1698$',
's16a4_4': r'$\mathbf{P}-\mathbf{Y} \;=\; (0.0420-0,\ 0.1142-1,\ 0.8438-0) \;=\; (0.0420,\ -0.8858,\ 0.8438)$',
# ---- 16B
's16b1_1': r'$p=0.8 \;\Longleftrightarrow\; \mathrm{odds}=\dfrac{0.8}{0.2}=4 \;\Longleftrightarrow\; z=b+wx=\ln 4$',
's16b1_1b': r'$x=\dfrac{\ln 4-b}{w}=\dfrac{1.3863+5.7703}{1.2823}=5.581$',
's16b2_1': r'$c_k \;=\; \dfrac{n}{K\,n_k}\,,\qquad c_0=\dfrac{1000}{2\cdot894}=0.559\,,\qquad c_1=\dfrac{1000}{2\cdot106}=4.717$',
's16b3_1': r'$C\,\mathbf{X}^{T}(\mathbf{p}-\mathbf{y})+(1-\rho)\,\mathbf{w}+\rho\,\mathbf{v}=\mathbf{0}\,,\qquad |C\,[\mathbf{X}^{T}(\mathbf{p}-\mathbf{y})]_j|\leq\rho\ \ \mathrm{if}\ w_j=0$',
# ---- 16C
's16c2_1': r'$P(\mathrm{separable}) \;=\; 2^{1-n}\sum_{k=0}^{p}\binom{n-1}{k} \;=\; 0.0350\qquad(n=100,\ p=40)$',
's16c3_1': r'$x_{\min}=60\,x_{\mathrm{h}} \;\Longrightarrow\; w_{\min}=\dfrac{w_{\mathrm{h}}}{60} \;\Longrightarrow\; \dfrac{1}{2}\,w_{\min}^2 \;=\; \dfrac{1}{3600}\cdot\dfrac{1}{2}\,w_{\mathrm{h}}^2$',
's16c3_2': r'$\dfrac{1}{2}\,(0.86781)^2 \;=\; 0.3765\,,\qquad \dfrac{1}{2}\,(0.02137)^2 \;=\; 0.000228$',
# ---- questions
's16q1_1': r'$z \;=\; \ln\dfrac{p}{1-p} \;=\; b+\mathbf{w}^{T}\mathbf{x}\ \in\ (-\infty,\infty)\qquad\Longleftrightarrow\qquad p \;=\; \dfrac{1}{1+e^{-z}} \;\in\; (0,1)$',
's16q2_1': r'$\dfrac{\partial\ell_i}{\partial w_j} \;=\; (p_i-y_i)\,x_{ij}\qquad\mathrm{against}\qquad \dfrac{\partial\,\frac{1}{2}(p_i-y_i)^2}{\partial w_j} \;=\; (p_i-y_i)\,p_i(1-p_i)\,x_{ij}$',
's16q3_1': r'$\lambda \;=\; \dfrac{1}{Cn}\,:\qquad \dfrac{1}{426}=0.002347\,,\qquad \dfrac{1}{4260}=0.000235\,,\qquad C_2=\dfrac{426}{4260}=0.1$',
}
from mkeq import matimg
matimg('s16a3_1b', [r'$\mathbf{H} \;=\; \dfrac{1}{4}\,\mathbf{X}^{T}\mathbf{X} \;=\;$', [['2','9'],['9','51']]], '.')
mkeq(E, '.'); print('solution equations ok:', len(E))
