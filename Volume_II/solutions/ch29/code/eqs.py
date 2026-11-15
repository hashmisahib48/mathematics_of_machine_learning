# Equation images of the Chapter 29 solutions. Run: python3 code/eqs.py  (writes sol/ch29/s29*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 29A.1
's29a1_1': [r'$P_{5}=\binom{5}{3}\,0.8^{3}\,0.2^{2}+\binom{5}{4}\,0.8^{4}\,0.2+0.8^{5}$',
             r'$=10\times0.512\times0.04+5\times0.4096\times0.2+0.32768$',
             r'$=0.2048+0.4096+0.3277=0.9421$'],
's29a1_2': r'$P_{3}=3p^{2}-2p^{3}=3\times0.64-2\times0.512=1.92-1.024=0.896$',
# ---- 29A.2
's29a2_1': r'$\bar{p}=\dfrac{0.60+0.55+0.05}{3}=\dfrac{1.20}{3}=0.40$',
's29a2_2': r'$\bar{p}_{w}=\dfrac{1\times0.60+1\times0.55+3\times0.05}{1+1+3}=\dfrac{1.30}{5}=0.26$',
's29a2_3': r'$-\ln0.60=0.5108\,,\quad -\ln0.55=0.5978\,,\quad -\ln0.05=2.9957$',
's29a2_4': r'$\dfrac{0.5108+0.5978+2.9957}{3}=1.3681$',
# ---- 29A.3
's29a3_1': r'$\bar{f}=\dfrac{4+7+10}{3}=7\,,\qquad (\bar{f}-y)^{2}=(7-6)^{2}=1$',
's29a3_2': r'$\dfrac{1}{3}\sum_{m}(f_{m}-y)^{2}=\dfrac{(4-6)^{2}+(7-6)^{2}+(10-6)^{2}}{3}=\dfrac{4+1+16}{3}=7$',
's29a3_3': r'$\dfrac{1}{3}\sum_{m}(f_{m}-\bar{f})^{2}=\dfrac{(-3)^{2}+0^{2}+3^{2}}{3}=\dfrac{18}{3}=6$',
's29a3_4': r'$1\;=\;7-6$',
# ---- 29A.4
's29a4_1': r'$\mathrm{width}(\mathbf{Z})=3+3+3=9\,,\qquad 9+13=22$',
's29a4_2': r'$M\,(K+1)+1=3\times(7+1)+1=25$',
# ---- questions
's29q2_1': r'$\left(\dfrac{e_{1}+e_{2}}{2}\right)^{2}<e_{1}^{2}\ \Longleftrightarrow\ -3\,e_{1}<e_{2}<e_{1}\qquad(e_{1}>0)$',
}
mkeq(E, '.'); print('solution equations ok:', len(E))
