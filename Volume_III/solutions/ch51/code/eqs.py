# Equation images of the Chapter 51 solutions. Run: python3 code/eqs.py  (writes sol/ch51/s51*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 51A.1
's51a1_1': r'$d(\mathbf{a},\mathbf{b})=\sqrt{\dfrac{3}{1}\,(2-4)^2}=\sqrt{12}=3.464$',
's51a1_2': r'$d(\mathbf{a},\mathbf{c})=\sqrt{\dfrac{3}{2}\,\left[(2-1)^2+(6-9)^2\right]}=\sqrt{15}=3.873$',
's51a1_3': r'$d(\mathbf{a},\mathbf{d})=\sqrt{\dfrac{3}{1}\,(6-5)^2}=\sqrt{3}=1.732$',
's51a1_4': r'$\hat{x}_{a2}=\dfrac{6+5}{2}=5.5$',
# ---- 51A.2
's51a2_1': [r'$\bar{x}_o=\dfrac{2+4+6+8}{4}=5$', r'$s_o^2=\dfrac{(-3)^2+(-1)^2+1^2+3^2}{4}=\dfrac{20}{4}=5$'],
's51a2_2': r'$\tilde{\mathbf{x}}=(2,\,4,\,6,\,8,\,5,\,5)\,,\qquad \bar{\tilde{x}}=\dfrac{30}{6}=5$',
's51a2_3': r'$\mathrm{Var}(\tilde{x})=\dfrac{20+0+0}{6}=\dfrac{10}{3}=3.333$',
's51a2_4': r'$\dfrac{\mathrm{Var}(\tilde{x})}{s_o^2}=\dfrac{10/3}{5}=\dfrac{2}{3}=\dfrac{n_o}{n}=1-q\,,\qquad q=\dfrac{2}{6}$',
# ---- questions
's51q2_1': r'$\mathrm{Var}(\tilde{x})=(1-q)\,s_o^2\,,\qquad \mathrm{corr}(\tilde{x},z)\approx\sqrt{1-q}\;\mathrm{corr}(x,z)$',
}
mkeq(E, '.'); print('solution equations ok:', len(E))
