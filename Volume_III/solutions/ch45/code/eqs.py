# Equation images of the Chapter 45 solutions. Run: python3 code/eqs.py  (writes sol/ch45/s45*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq, matimg
E = {
# ---- 45A.1
's45a1_1': r'$\mathbf{d}=\mathbf{u}-\mathbf{v}=(3-1,\ 1-1,\ 2-0,\ 0-2,\ 1-1,\ 2-0)=(2,\ 0,\ 2,\ -2,\ 0,\ 2)$',
's45a1_2': r'$\Vert\mathbf{d}\Vert^{2} \;=\; 4+0+4+4+0+4 \;=\; 16$',
's45a1_3': [r'$z_1 \;=\; d_1-d_3+d_5 \;=\; 2-2+0 \;=\; 0$',
            r'$z_2 \;=\; d_2-d_4-d_6 \;=\; 0+2-2 \;=\; 0$',
            r'$z_3 \;=\; -d_1+d_4+d_6 \;=\; -2-2+2 \;=\; -2$'],
's45a1_4': r'$q \;=\; \dfrac{\Vert\mathbf{R}\mathbf{d}\Vert^{2}}{\Vert\mathbf{d}\Vert^{2}} \;=\; \dfrac{0^2+0^2+(-2)^2}{16} \;=\; \dfrac{4}{16} \;=\; 0.25$',
's45a1_5': r'$\mathrm{Var}\,q \;=\; \dfrac{2+(s-3)\sum_l d_l^{4}/\Vert\mathbf{d}\Vert^{4}}{k} \;=\; \dfrac{2+0}{3} \;=\; \dfrac{2}{3}\,,\qquad \sigma_q=0.816$',
's45a1_6': r'$\dfrac{q-1}{\sigma_q} \;=\; \dfrac{0.25-1}{0.816} \;=\; -0.92$',
# ---- 45A.2
's45a2_1': [r'$\mathbf{C}=\left(k(0,1),\,k(1,1),\,k(2,1)\right)^{T}=\left(e^{-1/2},\ 1,\ e^{-1/2}\right)^{T}=(0.6065,\ 1,\ 0.6065)^{T}$',
            r'$W=k(1,1)=1$'],
's45a2_2': [r'$\hat{K}_{ij} \;=\; \dfrac{k(x_i,1)\,k(1,x_j)}{k(1,1)}$',
            r'$\hat{K}_{00}=e^{-1/2}\,e^{-1/2}=e^{-1}=0.3679\,,\qquad \hat{K}_{02}=e^{-1/2}\,e^{-1/2}=0.3679$'],
's45a2_5': r'$\mathrm{tr}(\mathbf{K}-\hat{\mathbf{K}}) \;=\; (1-e^{-1})+0+(1-e^{-1}) \;=\; 2\times0.6321 \;=\; 1.2642$',
's45a2_6': r'$\lambda\left(\mathbf{K}-\hat{\mathbf{K}}\right) \;=\; 0.6321\pm0.2325 \;=\; 0.8647\,,\ 0.3996\,,\ 0$',
# ---- 45C.1, 45C.2
's45c1_1': r'$k \;\geq\; \dfrac{4\ln n}{\varepsilon^{2}/2-\varepsilon^{3}/3} \;>\; \dfrac{4\ln 1347}{1/2-1/3} \;=\; \dfrac{28.82}{0.1667} \;=\; 172.9 \;>\; 64$',
's45c2_1': r'$\sigma_q \;=\; \sqrt{\dfrac{2+(s-3)\sum_l x_l^{4}/\Vert\mathbf{x}\Vert^{4}}{k}}$',
's45c2_2': [r'$\mathrm{Gaussian,\ both:}\quad \sigma_q=\sqrt{2/10}=0.447$',
            r'$\mathrm{sparse,\ flat:}\quad \sigma_q=\sqrt{(2+17/400)/10}=0.452$',
            r'$\mathrm{sparse,\ spiky:}\quad \sigma_q=\sqrt{(2+17)/10}=1.378$'],
# ---- questions
's45q4_1': r'$\mathbf{w}^{T}\mathbf{z}(\mathbf{x})+b \;=\; \mathbf{w}^{T}\mathbf{W}^{-1/2}\,\mathbf{k}_I(\mathbf{x})+b \;=\; \tilde{\mathbf{w}}^{T}\mathbf{k}_I(\mathbf{x})+b\,,\qquad \tilde{\mathbf{w}}=\mathbf{W}^{-1/2}\mathbf{w}$',
}
mkeq(E, '.')
f = lambda r: [[x for x in row] for row in r]
matimg('s45a2_3', [r'$\hat{\mathbf{K}}\;=\;\mathbf{C}\mathbf{C}^{T}\;=\;$', f([['e^{-1}','e^{-1/2}','e^{-1}'],['e^{-1/2}','1','e^{-1/2}'],['e^{-1}','e^{-1/2}','e^{-1}']]),
   r'$=$', f([['0.3679','0.6065','0.3679'],['0.6065','1','0.6065'],['0.3679','0.6065','0.3679']])], '.')
matimg('s45a2_4', [r'$\mathbf{K}-\hat{\mathbf{K}}\;=\;$', f([['0.6321','0','-0.2325'],['0','0','0'],['-0.2325','0','0.6321']])], '.')
print('solution equations ok:', len(E) + 2)
