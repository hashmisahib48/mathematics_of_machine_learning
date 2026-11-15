# Equation images of the Chapter 55 solutions. Run: python3 code/eqs.py  (writes sol/ch55/s55*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 55A.1
's55a1_1': [r'$a_{k0}=\#\{\,j:\ 0\leq j<9,\ j\equiv k-1\ (\mathrm{mod}\ 4)\,\}$',
             r'$a_{k1}=\#\{\,j:\ 9\leq j<14,\ j\equiv k-1\ (\mathrm{mod}\ 4)\,\}$'],
's55a1_2': r'$\lfloor 9/4\rfloor=2\,,\ \lceil 9/4\rceil=3\,;\qquad \lfloor 5/4\rfloor=1\,,\ \lceil 5/4\rceil=2$',
# ---- 55A.2
's55a2_1': r'$7+2=9\,,\qquad 5+3=8\,,\qquad 4+3+1=8\,,\qquad 9+8+8=25$',
# ---- question 1
's55q1_1': [r'$\mathrm{MSE}_{\mathrm{same}}=\sigma^{2}\left(1+\dfrac{1}{m-1}\right)=1\times\dfrac{8}{7}=1.14$',
            r'$\mathrm{MSE}_{\mathrm{new}}\approx\sigma_a^{2}+\sigma^{2}=4+1=5$'],
}
mkeq(E, '.'); print('solution equations ok:', len(E))
