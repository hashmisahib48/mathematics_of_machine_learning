# Equation images of the Chapter 25 solutions. Run: python3 code/eqs.py  (writes sol/ch25/s25*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 25A.1
's25a1_1': [r'$\left(\dfrac{7}{8}\right)^{2}=0.7656\,,\qquad \left(\dfrac{7}{8}\right)^{4}=0.7656^{2}=0.5862$',
            r'$\left(1-\dfrac{1}{8}\right)^{8}=\left(\dfrac{7}{8}\right)^{8}=0.5862^{2}=0.3436$'],
's25a1_2': r'$n\left[1-\left(1-\dfrac{1}{n}\right)^{n}\right] \;=\; 8\times(1-0.3436) \;=\; 8\times 0.6564 \;=\; 5.25$',
# ---- 25A.2
's25a2_1': [r'$\rho\sigma^{2}=0.4\times 9=3.6\,,\qquad (1-\rho)\sigma^{2}=0.6\times 9=5.4$',
            r'$\mathrm{Var}\left[\bar f\right]=\rho\sigma^{2}+\dfrac{(1-\rho)\sigma^{2}}{B}=3.6+\dfrac{5.4}{10}=3.6+0.54=4.14$'],
's25a2_2': r'$\lim_{B\to\infty}\mathrm{Var}\left[\bar f\right] \;=\; \rho\sigma^{2} \;=\; 3.6$',
's25a2_3': [r'$3.6+\dfrac{5.4}{B}\;\leq\;1.1\times 3.6=3.96$',
            r'$\dfrac{5.4}{B}\;\leq\;0.36\quad\Longrightarrow\quad B\;\geq\;\dfrac{5.4}{0.36}=15$'],
's25a2_4': r'$B=14:\; 3.6+\dfrac{5.4}{14}=3.9857\,,\qquad B=15:\; 3.6+\dfrac{5.4}{15}=3.9600$',
# ---- 25A.3
's25a3_1': [r'$\bar p_1=\dfrac{0.9+0.4+0.45}{3}=\dfrac{1.75}{3}=0.5833$',
            r'$\bar p_2=\dfrac{0.6+0.6+0.1}{3}=\dfrac{1.30}{3}=0.4333$'],
# ---- 25A.4
's25a4_1': r'$I(\mathrm{root}) \;=\; 1-\left(\dfrac{2}{4}\right)^{2}-\left(\dfrac{2}{4}\right)^{2} \;=\; 0.5$',
's25a4_2': [r'$s=2.5:\quad G_L=0\,,\quad G_R=1-\left(\dfrac{1}{3}\right)^{2}-\left(\dfrac{2}{3}\right)^{2}=0.4444$',
            r'$\Delta \;=\; 0.5-\dfrac{1}{4}\cdot 0-\dfrac{3}{4}\cdot 0.4444 \;=\; 0.5-0.3333 \;=\; 0.1667$'],
's25a4_3': r'$s=4:\quad G_L=G_R=0\,,\qquad \Delta \;=\; 0.5-0 \;=\; 0.5$',
's25a4_4': [r'$s=8:\quad G_L=1-\left(\dfrac{2}{3}\right)^{2}-\left(\dfrac{1}{3}\right)^{2}=0.4444\,,\quad G_R=0$',
            r'$\Delta \;=\; 0.5-\dfrac{3}{4}\cdot 0.4444-\dfrac{1}{4}\cdot 0 \;=\; 0.1667$'],
's25a4_5': r'$P(\Delta=0.5) \;=\; \dfrac{7-3}{9-2} \;=\; \dfrac{4}{7} \;=\; 0.5714$',
# ---- 25B
's25b1_1': r'$n\,(1-e^{-1})^{50} \;=\; 426\times 0.6321^{50} \;=\; 4.7\times 10^{-8}$',
's25b3_1': r'$w_0=\dfrac{1000}{2\times 921}=0.543\,,\qquad w_1=\dfrac{1000}{2\times 79}=6.33$',
# ---- 25C
's25c1_1': r'$0.143+0.140 \;=\; 0.283 \;\approx\; 0.280 \;(\mathrm{one\ copy})$',
's25c2_1': r'$\mathrm{Var}\left[\bar f_{B}\right] \;=\; \rho\sigma^{2}+\dfrac{(1-\rho)\sigma^{2}}{B}$',
's25c2_2': r'$\sqrt{\dfrac{0.05\times 0.95}{143}} \;=\; 0.0182\,,\qquad \dfrac{1}{143}=0.0070$',
# ---- questions
's25q1_1': r'$\mathrm{Var}\left[\bar f\right] \;=\; \rho\,\sigma^{2} \;+\; \dfrac{(1-\rho)\,\sigma^{2}}{B}$',
's25q4_1': r'$1-(1-0.632)^{4} \;=\; 1-0.368^{4} \;=\; 0.9817$',
}
mkeq(E, '.'); print('solution equations ok:', len(E))
