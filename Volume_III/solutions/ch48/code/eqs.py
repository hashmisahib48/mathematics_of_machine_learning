# Equation images of the Chapter 48 solutions. Run: python3 code/eqs.py  (writes sol/ch48/s48*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq, matimg
E = {
# ---- 48A.1
's48a1_1': [r'$c(16) \;=\; 2\,(\ln 15+\gamma_{E})-\dfrac{2\times 15}{16}$', r'$\;=\; 2\,(2.7081+0.5772)-1.875 \;=\; 6.5705-1.875 \;=\; 4.6955$'],
's48a1_2': [r'$\mathrm{E}\,h=2:\quad s=2^{-2/4.6955}=2^{-0.4259}=0.7444$',
            r'$\mathrm{E}\,h=4:\quad s=2^{-4/4.6955}=2^{-0.8519}=0.5541$',
            r'$\mathrm{E}\,h=6:\quad s=2^{-6/4.6955}=2^{-1.2778}=0.4124$'],
's48a1_3': r'$s>0.5 \;\Longleftrightarrow\; 2^{-\mathrm{E}h/c}>2^{-1} \;\Longleftrightarrow\; \mathrm{E}\,h<c(16)=4.6955$',
# ---- 48A.2
's48a2_1': r'$\mathrm{lrd}(10)=\dfrac{1}{(6+8)/2}=\dfrac{1}{7}=0.1429\,,\qquad \mathrm{lrd}(4)=\dfrac{1}{(2+3)/2}=\dfrac{2}{5}=0.4$',
's48a2_2': r'$\mathrm{LOF}(10)=\dfrac{\left(\mathrm{lrd}(4)+\mathrm{lrd}(2)\right)/2}{\mathrm{lrd}(10)}=\dfrac{(2/5+2/3)/2}{1/7}=\dfrac{8}{15}\times7=\dfrac{56}{15}=3.7333$',
's48a2_3': r'$\mathrm{LOF}(4)=\dfrac{\left(\mathrm{lrd}(2)+\mathrm{lrd}(1)\right)/2}{\mathrm{lrd}(4)}=\dfrac{(0.6667+0.5)/2}{0.4}=\dfrac{0.5833}{0.4}=1.4583$',
# ---- 48A.3
's48a3_2': r'$\det\hat{\mathbf{\Sigma}} \;=\; 16\times0.25-1.2^{2} \;=\; 4-1.44 \;=\; 2.56$',
's48a3_3': [r'$d^{2}=\mathbf{d}^{T}\hat{\mathbf{\Sigma}}^{-1}\mathbf{d}=64\times0.0977+2\times8\times1\times(-0.4688)+1\times6.25$', r'$\;=\;6.25-7.5+6.25 \;=\; 5.00$'],
's48a3_4': [r'$\rho=\dfrac{1.2}{4\times0.5}=0.6\,,\quad z_1=\dfrac{8}{4}=2\,,\quad z_2=\dfrac{1}{0.5}=2$',
            r'$d^{2}=\dfrac{z_1^{2}-2\rho\,z_1z_2+z_2^{2}}{1-\rho^{2}}=\dfrac{4-4.8+4}{0.64}=\dfrac{3.2}{0.64}=5.00$'],
's48a3_5': r'$\chi^{2}_{2;\,0.975} \;=\; -2\ln(1-0.975) \;=\; -2\ln0.025 \;=\; 7.3778$',
's48a3_6': r'$\Vert\mathbf{x}-\hat{\boldsymbol{\mu}}\Vert^{2} \;=\; 8^{2}+1^{2} \;=\; 65$',
# ---- 48B.1
's48b1_1': r'$\rho(t) \;\approx\; \dfrac{1}{1+c/t}\,,\qquad c \;=\; t\left(\dfrac{1}{\rho}-1\right)$',
# ---- 48B.3
's48b3_1': r'$\mathrm{precision} \;=\; \dfrac{15}{\gamma\,n}\,,\qquad \mathrm{recall}=1\quad(\gamma\,n\geq15)$',
# ---- 48C.2
's48c2_1': r'$\mathrm{precision} \;\leq\; \dfrac{20}{0.05\times10\,000} \;=\; \dfrac{20}{500} \;=\; 0.04$',
}
mkeq(E, '.')
matimg('s48a3_1', [r'$\hat{\mathbf{\Sigma}}^{-1}=\dfrac{1}{2.56}$', [['0.25', '-1.2'], ['-1.2', '16']], r'$=$', [['0.0977', '-0.4688'], ['-0.4688', '6.25']]], '.')
print('solution equations ok:', len(E) + 1)
