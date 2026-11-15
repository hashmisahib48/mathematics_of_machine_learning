# Equation images of the Chapter 58 solutions. Run: python3 code/eqs.py  (writes sol/ch58/s58*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 58A.1
's58a1_1': r'$t^{*}=\dfrac{c_{FP}}{c_{FP}+c_{FN}}=\dfrac{1}{1+19}=\dfrac{1}{20}=0.05$',
's58a1_2': r'$\ln\dfrac{c_{FP}}{c_{FN}}=\ln\dfrac{1}{19}=-2.9444$',
's58a1_3': r'$\mathrm{yes}:\ 0.96\times 1=0.96\,,\qquad \mathrm{no}:\ 0.04\times 19=0.76$',
's58a1_4': r'$\mathrm{yes}:\ 0.94\times 1=0.94\,,\qquad \mathrm{no}:\ 0.06\times 19=1.14$',
# ---- 58A.2
's58a2_1': r'$s(f;\mathbf{X},\mathbf{y})=\dfrac{6}{8}=0.75$',
's58a2_2': r'$\dfrac{1}{n}\sum_i\dfrac{m_{y_i}}{n}=\dfrac{1}{8}\left(3\times\dfrac{5}{8}+5\times\dfrac{3}{8}\right)=\dfrac{1}{8}\times 3.75=0.46875$',
's58a2_3': r'$E[\mathrm{accuracy}]=\dfrac{n_1m_1+n_0m_0}{n^{2}}=\dfrac{3\times 5+5\times 3}{8^{2}}=\dfrac{30}{64}=0.46875$',
's58a2_4': r'$I=0.75-0.46875=0.28125\approx 0.281$',
# ---- 58C.2
's58c2_1': r'$\mathrm{PD}(v)=\dfrac{1}{2}\,f(v,+)+\dfrac{1}{2}\,f(v,-)=\dfrac{1}{2}\cdot 1+\dfrac{1}{2}\cdot 0=\dfrac{1}{2}$',
's58c2_2': r'$\mathrm{PD}(v)=0.49\,\bar{c}_{+}(v)+0.51\,\bar{c}_{-}(v)$',
# ---- questions
's58q1_1': r'$t^{*}=\dfrac{c_{FP}}{c_{FP}+c_{FN}}=\dfrac{1}{1+10}=\dfrac{1}{11}=0.0909$',
}
mkeq(E, '.')
from PIL import Image
for k in E:
    with Image.open(k + '.png') as im: w = im.size[0] / 300 * 96
    if w > 400: print('TOO WIDE', k, round(w))
