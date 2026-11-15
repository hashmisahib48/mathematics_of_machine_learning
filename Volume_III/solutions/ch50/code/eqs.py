# Equation images of the Chapter 50 solutions. Run: python3 code/eqs.py  (writes sol/ch50/s50*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq, matimg
E = {
# ---- 50A.1
's50a1_3': [r'$\mathbf{D}^{T}\mathbf{y} \;=\; (4+6,\ 10,\ 3+5+7)^{T} \;=\; (10,\ 10,\ 15)^{T}$',
             r'$\mathbf{w} \;=\; (\mathbf{D}^{T}\mathbf{D})^{-1}\mathbf{D}^{T}\mathbf{y} \;=\; \left(\dfrac{10}{2},\ \dfrac{10}{1},\ \dfrac{15}{3}\right) \;=\; (5,\ 10,\ 5)$'],
's50a1_4': r'$\mathbf{1}-\mathbf{d}_a-\mathbf{d}_b-\mathbf{d}_c \;=\; \mathbf{0}\quad\Longrightarrow\quad \mathrm{rank}\,[\,\mathbf{1}\,|\,\mathbf{D}\,] \;=\; 3\;<\;4\ \mathrm{columns}$',
's50a1_6': r'$\boldsymbol{\beta} \;=\; (\beta_0,\ \beta_b,\ \beta_c) \;=\; (\bar{y}_a,\ \bar{y}_b-\bar{y}_a,\ \bar{y}_c-\bar{y}_a) \;=\; (5,\ 5,\ 0)$',
# ---- 50A.2
's50a2_1': r'$\lambda_c \;=\; \dfrac{n_c}{n_c+m}\,:\qquad \lambda_1=\dfrac{1}{5}=0.2\,,\quad \lambda_2=\dfrac{4}{8}=0.5\,,\quad \lambda_3=\dfrac{20}{24}=0.8333$',
's50a2_2': [r'$t_1 \;=\; 0.2\times 30+0.8\times 14 \;=\; 6+11.2 \;=\; 17.2$',
            r'$t_2 \;=\; 0.5\times 20+0.5\times 14 \;=\; 10+7 \;=\; 17$',
            r'$t_3 \;=\; 0.8333\times 10+0.1667\times 14 \;=\; 8.3333+2.3333 \;=\; 10.6667$'],
's50a2_3': [r'$t_c-\bar{y}_c \;=\; (1-\lambda_c)\,(\bar{y}-\bar{y}_c)$',
             r'$0.8\times(-16)=-12.8\,,\qquad 0.5\times(-6)=-3\,,\qquad 0.1667\times 4=0.667$'],
# ---- 50B.2
's50b2_1': r'$Z_i \;=\; \dfrac{\sum_{j\in c}y_j+m\,\bar{y}}{r+m}\,,\qquad \mathrm{Cov}(Z_i,y_i)=\dfrac{\sigma^2}{r+m}\,,\quad \mathrm{Var}(Z_i)=\dfrac{r\,\sigma^2}{(r+m)^2}$',
's50b2_2': r'$\rho \;=\; \dfrac{\sigma^2/(r+m)}{\sigma\cdot\sigma\sqrt{r}/(r+m)} \;=\; \dfrac{1}{\sqrt{r}}$',
# ---- 50C.1
's50c1_1': r'$\sum_{k} n_k\,(\bar{y}_k-b_0-w_k)^2 \;+\; \alpha\sum_{k} w_k^2 \;\;\Longrightarrow\;\; w_k \;=\; \dfrac{n_k\,(\bar{y}_k-b_0)}{n_k+\alpha}$',
's50c1_1b': r'$b_0 \;=\; \dfrac{\sum_k \omega_k\,\bar{y}_k}{\sum_k \omega_k}\,,\qquad \omega_k \;=\; \dfrac{n_k\,\alpha}{n_k+\alpha}$',
's50c1_2': r'$\mathrm{all\ three:}\quad b_0 \;=\; \dfrac{0.6667\times 5+0.5\times 10+0.75\times 5}{0.6667+0.5+0.75} \;=\; \dfrac{12.083}{1.9167} \;=\; 6.3043$',
's50c1_2b': r'$\mathrm{a\ dropped:}\quad b_0 \;=\; \dfrac{2\times 5+0.5\times 10+0.75\times 5}{2+0.5+0.75} \;=\; \dfrac{18.75}{3.25} \;=\; 5.7692$',
# ---- 50C.2
's50c2_1': r'$\mathbf{z}_h \;=\; \left(\sin\dfrac{2\pi h}{24},\ \cos\dfrac{2\pi h}{24}\right)\,,\qquad \Vert\mathbf{z}_{23}-\mathbf{z}_{0}\Vert \;=\; 2\sin\dfrac{\pi}{24} \;=\; 0.261 \;=\; \Vert\mathbf{z}_{12}-\mathbf{z}_{11}\Vert$',
# ---- questions
's50q1_1': r'$\mathbf{z}_m \;=\; \left(\sin\dfrac{2\pi m}{12},\ \cos\dfrac{2\pi m}{12}\right)\,,\qquad \Vert\mathbf{z}_{12}-\mathbf{z}_{1}\Vert \;=\; 2\sin\dfrac{\pi}{12} \;=\; 0.518 \;=\; \Vert\mathbf{z}_{2}-\mathbf{z}_{1}\Vert$',
}
mkeq(E, '.')
matimg('s50a1_1', [r'$\mathbf{D}=$', [['1','0','0'],['1','0','0'],['0','1','0'],['0','0','1'],['0','0','1'],['0','0','1']], r'$,\qquad \mathbf{y}=$', [['4'],['6'],['10'],['3'],['5'],['7']]])
matimg('s50a1_2', [r'$\mathbf{D}^{T}\mathbf{D}=$', [['2','0','0'],['0','1','0'],['0','0','3']], r'$=\ \mathrm{diag}(n_a,\,n_b,\,n_c)$'])
matimg('s50a1_5', [r'$\mathbf{B}^{T}\mathbf{B}=\left[\mathbf{1}\,|\,\mathbf{d}_b\,|\,\mathbf{d}_c\right]^{T}\left[\mathbf{1}\,|\,\mathbf{d}_b\,|\,\mathbf{d}_c\right]=$', [['6','1','3'],['1','1','0'],['3','0','3']], r'$,\qquad \det\mathbf{B}^{T}\mathbf{B}=6\neq 0$'])
print('solution equations ok:', len(E) + 3)
