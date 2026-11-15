# Equation images of the Chapter 56 solutions. Run: python3 code/eqs.py  (writes sol/ch56/s56*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 56A.1
's56a1_1': r'$n_{\mathrm{req}} \;=\; 1+\lfloor\log_3 100\rfloor \;=\; 1+\lfloor 4.19\rfloor \;=\; 5$',
's56a1_2': r'$r_0 \;=\; \max\left(20\,,\ \lfloor 1800/3^{4}\rfloor\right) \;=\; \max(20,\,22) \;=\; 22$',
's56a1_3': r'$n_{\mathrm{poss}} \;=\; 1+\lfloor\log_3\lfloor 1800/22\rfloor\rfloor \;=\; 1+\lfloor\log_3 81\rfloor \;=\; 5$',
's56a1_4': [r'$\sum_k n_k r_k \;=\; 2200+2244+2376+2376+3564 \;=\; 12760$',
            r'$n_0 R \;=\; 100\times 1800 \;=\; 180000\,,\qquad 180000/12760 \;=\; 14.1$'],
# ---- 56A.2
's56a2_1': r'$n \;\geq\; \dfrac{\ln(1-P)}{\ln(1-q)} \;=\; \dfrac{\ln 0.01}{\ln 0.98} \;=\; \dfrac{-4.6052}{-0.02020} \;=\; 227.95$',
's56a2_2': r'$P \;=\; 1-0.98^{60} \;=\; 1-0.2976 \;=\; 0.7024$',
's56a2_3': [r'$q=0.05:\quad n \;\geq\; \dfrac{-4.6052}{-0.05129} \;=\; 89.78 \;\Rightarrow\; 90$',
            r'$q=0.10:\quad n \;\geq\; \dfrac{-4.6052}{-0.10536} \;=\; 43.71 \;\Rightarrow\; 44$'],
}
mkeq(E, '.'); print('solution equations ok:', len(E))
