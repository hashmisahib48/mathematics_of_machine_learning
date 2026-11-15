# Equation images of the Chapter 7 solutions. Run: python3 code/eqs.py  (writes sol/ch07/s7*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 7A.1
's7a1_1': r'$\mathrm{data}=(5,\,6,\,8,\,1,\,3)\,,\qquad \mathrm{indices}=(0,\,3,\,2,\,1,\,4)$',
's7a1_2': r'$\mathrm{indptr} \;=\; (0,\ 0+2,\ 2+1,\ 3+2) \;=\; (0,\,2,\,3,\,5)$',
's7a1_3': r'$\mathrm{indptr}[2]=3 \;\leq\; k \;<\; \mathrm{indptr}[3]=5\,, \qquad k=3,\ 4$',
's7a1_5': r'$\dfrac{5\ \mathrm{stored}}{3\times 5\ \mathrm{entries}} \;=\; \dfrac{5}{15} \;=\; \dfrac{1}{3}$',
# ---- 7A.2
's7a2_1': r'$\mu \;=\; \dfrac{0+2+2+8}{4} \;=\; \dfrac{12}{4} \;=\; 3$',
's7a2_2': [r'$\sigma^2 \;=\; \dfrac{(-3)^2+(-1)^2+(-1)^2+5^2}{4} \;=\; \dfrac{9+1+1+25}{4}$', r'$=\; \dfrac{36}{4} \;=\; 9\,,\qquad \sigma=3$'],
's7a2_3': r'$z \;=\; \dfrac{12-3}{3} \;=\; 3$',
's7a2_4': r'$\sigma_{n-1} \;=\; \sqrt{\dfrac{36}{4-1}} \;=\; \sqrt{12} \;=\; 3.4641 \;=\; 3\sqrt{\dfrac{4}{3}}$',
# ---- 7A.3
's7a3_3': r'$\dfrac{e^{f_k+5}}{\sum_j e^{f_j+5}} \;=\; \dfrac{e^{5}\,e^{f_k}}{e^{5}\sum_j e^{f_j}} \;=\; \dfrac{e^{f_k}}{\sum_j e^{f_j}} \;=\; P_k$',
# ---- 7A.3 (one fraction)
's7a3_2': r'$P_1 \;=\; \dfrac{7.389}{8.757} \;=\; 0.844$',
# ---- 7A.4
's7a4_1': r'$n_{\mathrm{test}} \;=\; \lceil f\,n\rceil\,,\qquad n_{\mathrm{train}} \;=\; n - n_{\mathrm{test}}$',
# ---- 7C.3
's7c3_1': r'$s_{A}=\sqrt{\dfrac{0.9737\times 0.0263}{38}}=0.0260$',
's7c3_3': r'$s_{B}=\sqrt{\dfrac{0.9474\times 0.0526}{38}}=0.0362$',
's7c3_2': r'$s_{A-B} \;=\; \sqrt{s_A^{2}+s_B^{2}} \;=\; 0.0446\,,\qquad \dfrac{0.9737-0.9474}{0.0446} \;=\; 0.59$',
# ---- questions
's7q4_1': r'$s \;=\; \sqrt{\dfrac{\dfrac{37}{38}\times\dfrac{1}{38}}{38}} \;=\; 0.0260$',
}
mkeq(E, '.'); print('solution equations ok:', len(E))
