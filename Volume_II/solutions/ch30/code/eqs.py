# Equation images of the Chapter 30 solutions. Run: python3 code/eqs.py  (writes sol/ch30/s30*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq, matimg
E = {
# ---- 30A.1
's30a1_2': [r'$k(0,1)=e^{-1^{2}/2}=e^{-0.5}=0.6065\,,\qquad k(1,3)=e^{-2^{2}/2}=e^{-2}=0.1353$',
             r'$k(0,3)=e^{-3^{2}/2}=e^{-4.5}=0.0111$'],
's30a1_3': r'$\mathrm{corr}\,[f(0),f(3)]=\dfrac{k(0,3)}{\sqrt{k(0,0)\,k(3,3)}}=\dfrac{0.0111}{\sqrt{1\times1}}=0.0111$',
's30a1_4': r'$k(0,3)=\exp\!\left(-\dfrac{3^{2}}{2\times2^{2}}\right)=e^{-9/8}=e^{-1.125}=0.3247$',
# ---- 30A.2
's30a2_1': [r'$m=\mathbf{k}_*^{T}\mathbf{S}^{-1}\mathbf{y}=\dfrac{0.8}{1}\times1.5=1.2$',
            r'$v=k_{**}-\mathbf{k}_*^{T}\mathbf{S}^{-1}\mathbf{k}_*=1-\dfrac{0.8^{2}}{1}=0.36$'],
's30a2_2': [r'$m=\dfrac{0.8}{1.25}\times1.5=0.64\times1.5=0.96$',
            r'$v=1-\dfrac{0.8^{2}}{1.25}=1-0.512=0.488$'],
's30a2_3': r'$m=\dfrac{0.8}{1+\sigma^{2}}\times1.5\;\longrightarrow\;0\qquad(\sigma^{2}\rightarrow\infty)$',
's30a2_4': r'$v=1-\dfrac{0.64}{1+\sigma^{2}}\;\longrightarrow\;1\qquad(\sigma^{2}\rightarrow\infty)$',
# ---- 30A.3
's30a3_1': [r'$S=k(0,0)+\sigma^{2}=1+0.25=1.25$', r'$a=S^{-1}y=\dfrac{1.5}{1.25}=1.2$'],
's30a3_2': [r'$m(x_*)=k_*\,a=1.2\,e^{-x_*^{2}/2}$', r'$v(x_*)=1-\dfrac{k_*^{2}}{S}=1-\dfrac{e^{-x_*^{2}}}{1.25}$'],
's30a3_3': [r'$m(0)=1\times1.2=1.2\,,\qquad v(0)=1-\dfrac{1}{1.25}=0.2$'],
# ---- 30C.1
's30c1_1': r'$\ln p(\mathbf{y})=-\frac{1}{2}\,\mathbf{y}^{T}\mathbf{S}^{-1}\mathbf{y}\;-\;\frac{1}{2}\ln|\mathbf{S}|\;-\;\frac{n}{2}\ln 2\pi$',
's30c1_2': [r'$\mathrm{smooth:}\quad -\frac{1}{2}(1.06)+3.59-5.51=-2.45$', r'$\mathrm{zigzag:}\quad -\frac{1}{2}(122.91)+3.59-5.51=-63.38$'],
# ---- 30C.3
's30c3_1': [r'$\mathrm{memory}=8\,n^{2}=8\times(2\times10^{4})^{2}\ \mathrm{bytes}=3.2\times10^{9}\ \mathrm{bytes}$',
            r'$t(n)\approx t(2000)\left(\dfrac{n}{2000}\right)^{3}\,,\qquad \left(\dfrac{20000}{2000}\right)^{3}=1000$'],
# ---- questions
's30q4_1': r'$v(\mathbf{x}_*)=k_{**}-\mathbf{k}_*^{T}\,\mathbf{S}^{-1}\,\mathbf{k}_*$',
's30q5_1': r'$p(\mathbf{f}\mid\mathbf{y})\;\approx\;\mathcal{N}\!\left(\hat{\mathbf{f}},\;(\mathbf{K}^{-1}+\mathbf{W})^{-1}\right)$',
}
mkeq(E, '.')
r, g = 'e^{-1/2}', 'e^{-2}'
matimg('s30a1_1', [r'$\mathbf{K}=$', [['1', r'e^{-1/2}', r'e^{-9/2}'], [r'e^{-1/2}', '1', r'e^{-2}'], [r'e^{-9/2}', r'e^{-2}', '1']],
                   r'$=$', [['1', '0.6065', '0.0111'], ['0.6065', '1', '0.1353'], ['0.0111', '0.1353', '1']]], '.')
print('solution equations ok:', len(E) + 1)
