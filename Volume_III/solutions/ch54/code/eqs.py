# Equation images of the Chapter 54 solutions. Run: python3 code/eqs.py  (writes sol/ch54/s54*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 54A.1
's54a1_1': r'$\mathrm{median}(y\,|\,\mathbf{x}) \;=\; e^{\hat{\mu}(\mathbf{x})} \;=\; e^{2.0} \;=\; 7.389$',
's54a1_2': r'$e^{\sigma^{2}/2} \;=\; e^{0.25/2} \;=\; e^{0.125} \;=\; 1.1331$',
's54a1_3': r'$\mathrm{E}(y\,|\,\mathbf{x}) \;=\; e^{\hat{\mu}+\sigma^{2}/2} \;=\; 7.389\times 1.1331 \;=\; 8.373 \;=\; e^{2.125}$',
's54a1_4': r'$\dfrac{8.373-7.389}{7.389} \;=\; e^{0.125}-1 \;=\; 0.1331 \;=\; 13.31\ \%$',
# ---- 54A.2
's54a2_1': r'$\mathrm{(a)}\quad 6+4+1 \;=\; 11$',
's54a2_2': r'$\mathrm{(b)}\quad 11+3 \;=\; 14$',
's54a2_3': r'$\mathrm{(c)}\quad (6-1)+(4-1)+1+3 \;=\; 5+3+1+3 \;=\; 12$',
# ---- questions
's54q4_1': r'$\sum_{i}\mathrm{E}(y_i\,|\,\mathbf{x}_i) \;=\; \sum_{i} e^{\mu(\mathbf{x}_i)}\,e^{\sigma^{2}/2} \;\neq\; \sum_{i} e^{\mu(\mathbf{x}_i)}$',
}
mkeq(E, '.')
