# Equation images of the Chapter 6 solutions. Run: python3 code/eqs.py  (writes sol/ch06/s6*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 6A.1
's6a1_1': [r'$J_{\mathrm{train}} \,=\, \dfrac{1}{4}\left[(1-1)^2+(2-2)^2+(2-3)^2+(4-4)^2\right]$', r'$=\, \dfrac{0+0+1+0}{4} \,=\, 0.25$'],
's6a1_2': [r'$J_{\mathrm{test}} \,=\, \dfrac{1}{3}\left[(6-5)^2+(5-6)^2+(8-7)^2\right]$', r'$=\, \dfrac{1+1+1}{3} \,=\, 1.00$'],
's6a1_3': r'$\mathrm{gap} \;=\; J_{\mathrm{test}}-J_{\mathrm{train}} \;=\; 1.00-0.25 \;=\; 0.75$',
# ---- 6A.2
's6a2_1': [r'$E[\hat f] \,=\, \dfrac{2+4+4+6}{4} \,=\, 4\,,\qquad \mathrm{bias} \,=\, 4-5 \,=\, -1$', r'$\mathrm{bias}^2 \,=\, 1$',
           r'$\mathrm{Var} \,=\, \dfrac{(2-4)^2+0+0+(6-4)^2}{4} \,=\, \dfrac{8}{4} \,=\, 2$'],
's6a2_2': r'$\sigma^2+\mathrm{bias}^2+\mathrm{Var} \;=\; 0.5+1+2 \;=\; 3.5$',
's6a2_3': r'$\sigma^2+\dfrac{1}{4}\sum\,(f-\hat f)^2 \;=\; 0.5+\dfrac{9+1+1+1}{4} \;=\; 0.5+3 \;=\; 3.5$',
's6a2_4': [r'$E[\hat f] \,=\, \dfrac{3.5+4+4+4.5}{4} \,=\, 4\,,\qquad \mathrm{bias}^2 \,=\, (4-5)^2 \,=\, 1$',
           r'$\mathrm{Var} \,=\, \dfrac{(-0.5)^2+0+0+0.5^2}{4} \,=\, 0.125$'],
's6a2_5': r'$\sigma^2+\mathrm{bias}^2+\mathrm{Var} \;=\; 0.5+1+0.125 \;=\; 1.625$',
's6a2_6': r'$0.5+\dfrac{1.5^2+1+1+0.5^2}{4} \;=\; 0.5+1.125 \;=\; 1.625$',
# ---- 6A.3
's6a3_1': r'$E[J_{\mathrm{train}}] \;=\; \sigma^2\left(1-\dfrac{p}{n}\right) \;=\; 0.09\times\left(1-\dfrac{6}{30}\right) \;=\; 0.072$',
's6a3_2': r'$E[J_{\mathrm{test}}] \;=\; \sigma^2\left(1+\dfrac{p}{n}\right) \;=\; 0.09\times 1.2 \;=\; 0.108$',
's6a3_3': r'$\mathrm{gap} \;=\; \dfrac{2p\sigma^2}{n} \;=\; \dfrac{2\times 6\times 0.09}{30} \;=\; 0.036 \;=\; 0.108-0.072$',
's6a3_4': r'$n \;=\; \dfrac{2p\sigma^2}{\mathrm{gap}} \;=\; \dfrac{2\times 6\times 0.09}{0.006} \;=\; 180$',
# ---- 6A.4
's6a4_1': [r'$S_{xy}=(-2)(-3)+(-1)(-1)+1\cdot 1+2\cdot 3=14$', r'$S_{xx}=4+1+1+4=10$'],
's6a4_2': [r'$\hat w_{\alpha} \;=\; \dfrac{S_{xy}}{S_{xx}+\alpha}$', r'$\hat w_0=\dfrac{14}{10}=1.4\,,\quad \hat w_4=\dfrac{14}{14}=1.0\,,\quad \hat w_{10}=\dfrac{14}{20}=0.7$'],
's6a4_3': [r'$J_{\mathrm{train}} \,=\, \dfrac{(-3+2)^2+(-1+1)^2+(1-1)^2+(3-2)^2}{4}$', r'$=\, \dfrac{1+0+0+1}{4} \,=\, 0.5$'],
's6a4_4': r'$\dfrac{14}{10+\alpha} \;=\; 0.5 \quad\Longrightarrow\quad 10+\alpha \;=\; 28 \quad\Longrightarrow\quad \alpha \;=\; 18$',
# ---- 6C.3
's6c3_1': r'$E[J_{\mathrm{test}}] \;=\; \sigma^2\left(1+\dfrac{p}{n}\right)\ \longrightarrow\ \sigma^2 \;=\; 0.0625\quad(n\to\infty)$',
# ---- questions
's6q2_1': r'$E[(y-\hat f)^2]\;=\;\sigma^2 \;+\; \mathrm{bias}_d^2 \;+\; \dfrac{\sigma^2}{n}\,V_d\,,\qquad V_d \ \mathrm{increasing\ in}\ d$',
's6q4_1': r'$\hat w_{\alpha} \;=\; \dfrac{S_{xy}}{S_{xx}+\alpha} \;=\; \dfrac{1}{1+\alpha/S_{xx}}\;\hat w_{0}$',
}
mkeq(E, '.'); print('solution equations ok:', len(E))
