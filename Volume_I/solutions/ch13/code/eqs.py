# Equation images of the Chapter 13 solutions. Run: python3 code/eqs.py  (writes sol/ch13/s13*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 13A.1
's13a1_1': r'$\epsilon\,\sigma \;=\; 1.35 \times 2 \;=\; 2.7$',
's13a1_2': r'$\omega_3=\dfrac{2.7}{3}=0.9\,,\qquad \omega_4=\dfrac{2.7}{6}=0.45\,,\qquad \omega_5=\dfrac{2.7}{12}=0.225$',
's13a1_3': r'$\omega_i\,r_i \;=\; -1,\ \ 2,\ \ 2.7,\ \ -2.7,\ \ 2.7$',
# ---- 13A.2
's13a2_1': r'$f^{\,s} \;=\; 0.6^{3} \;=\; 0.216\,,\qquad 1-f^{\,s} \;=\; 0.784$',
's13a2_2': r'$N \;=\; \left\lceil \dfrac{\ln 0.01}{\ln 0.784} \right\rceil \;=\; \left\lceil \dfrac{-4.6052}{-0.2433} \right\rceil \;=\; \lceil 18.92 \rceil \;=\; 19$',
's13a2_3': [r'$f^{\,s} \;=\; 0.6^{2} \;=\; 0.36\,,\qquad 1-f^{\,s} \;=\; 0.64$',
            r'$N \;=\; \left\lceil \dfrac{-4.6052}{-0.4463} \right\rceil \;=\; \lceil 10.32 \rceil \;=\; 11$'],
's13a2_4': r'$N \;=\; \left\lceil \dfrac{\ln 0.001}{\ln 0.784} \right\rceil \;=\; \left\lceil \dfrac{-6.9078}{-0.2433} \right\rceil \;=\; \lceil 28.39 \rceil \;=\; 29$',
's13a2_5': r'$0.784^{18} = 0.0125 \;>\; 0.01\,,\qquad 0.784^{19} = 0.0098 \;\leq\; 0.01$',
# ---- 13A.3
's13a3_1': r'$w_{ij} \;=\; \dfrac{y_j-y_i}{x_j-x_i}\,,\qquad \mathrm{e.g.}\ \ w_{15} \;=\; \dfrac{30-1}{4-0} \;=\; 7.25$',
's13a3_2': r'$\hat{w} \;=\; \mathrm{median}\{2,2,2,2,2,2,7.25,9,12.5,23\} \;=\; \dfrac{2+2}{2} \;=\; 2$',
's13a3_3': r'$y_i-2x_i \;=\; 1,\ 1,\ 1,\ 1,\ 22 \qquad\Longrightarrow\qquad \hat{b} \;=\; 1$',
's13a3_4': [r'$\bar{x}=2\,,\qquad \bar{y}=9.2\,,\qquad S_{xy}=62\,,\qquad S_{xx}=10$',
            r'$w=\dfrac{62}{10}=6.2\,,\qquad b=9.2-6.2\times2=-3.2$'],
's13a3_5': [r'$\mathrm{Theil\!-\!Sen:}\qquad \hat{y}(5) \;=\; 1+2\times5 \;=\; 11$',
            r'$\mathrm{Least\ squares:}\qquad \hat{y}(5) \;=\; -3.2+6.2\times5 \;=\; 27.8$'],
# ---- 13A.4
's13a4_1': r'$c=2:\qquad \sum\rho_{0.25} \;=\; 0.75+0+0.25+0.5+2 \;=\; 3.50$',
's13a4_2': r'$c=3:\qquad \sum\rho_{0.25} \;=\; 1.5+0.75+0+0.25+1.75 \;=\; 4.25$',
's13a4_3': [r'$n\,\dfrac{dJ}{dc} \;=\; (1-\tau)\,N_{\mathrm{below}} - \tau\,N_{\mathrm{above}}$',
            r'$c\ \mathrm{just\ below\ }2:\qquad 0.75\times1-0.25\times4=-0.25$',
            r'$c\ \mathrm{just\ above\ }2:\qquad 0.75\times2-0.25\times3=+0.75$'],
# ---- 13B.2
's13b2_1': r'$\alpha_1=0.2\,,\qquad \alpha_2=0.5\,,\qquad \alpha_3=0.6$',
# ---- 13C.1
's13c1_1': [r'$\epsilon=1.35:\quad n-k\epsilon^{2} \;=\; 5-2\times1.35^{2} \;=\; 1.355$',
            r'$\epsilon=1.40:\quad n-k\epsilon^{2} \;=\; 5-2\times1.40^{2} \;=\; 1.080$'],
# ---- questions
's13q3_1': r'$\dfrac{\partial\,\hat{w}}{\partial y_i} \;=\; \dfrac{x_i-\bar{x}}{S_{xx}} \;=\; -0.2,\ -0.1,\ 0,\ 0.1,\ 0.2$',
}
mkeq(E, '.'); print('solution equations ok:', len(E))
