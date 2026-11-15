# Equation images of the Chapter 22 solutions. Run: python3 code/eqs.py  (writes sol/ch22/s22*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 22A.1
's22a1_1': r'$f(4,3) \;=\; 3\times4+4\times3-10 \;=\; 14\,,\qquad \Vert\mathbf{w}\Vert \;=\; \sqrt{3^2+4^2} \;=\; 5$',
's22a1_2': r'$d \;=\; \dfrac{f(4,3)}{\Vert\mathbf{w}\Vert} \;=\; \dfrac{14}{5} \;=\; 2.8$',
's22a1_3': r'$f(0,0) \;=\; -10\,,\qquad d \;=\; \dfrac{\vert-10\vert}{5} \;=\; 2.0$',
's22a1_4': r'$\dfrac{2}{\Vert\mathbf{w}\Vert} \;=\; \dfrac{2}{5} \;=\; 0.4\,,\qquad \dfrac{1}{\Vert\mathbf{w}\Vert} \;=\; 0.2 \;\;\mathrm{on\ each\ side\ of}\;\; f=0$',
# ---- 22A.2
's22a2_1': r'$\mathbf{w}^{T}\mathbf{x}_1+b \;=\; -1\,,\qquad \mathbf{w}^{T}\mathbf{x}_2+b \;=\; +1$',
's22a2_2': r'$\mathbf{w} \;=\; \alpha_1y_1\mathbf{x}_1+\alpha_2y_2\mathbf{x}_2 \;=\; \alpha_2\mathbf{x}_2-\alpha_1\mathbf{x}_1\,,\qquad \alpha_1y_1+\alpha_2y_2=0 \;\Rightarrow\; \alpha_1=\alpha_2=\alpha$',
's22a2_3': r'$\mathbf{w} \;=\; \alpha\,(\mathbf{x}_2-\mathbf{x}_1) \;=\; \alpha\,(2,\,0)\,,\qquad \mathbf{w}^{T}(\mathbf{x}_2-\mathbf{x}_1) \;=\; 2 \;\Rightarrow\; 4\alpha=2 \;\Rightarrow\; \alpha=0.5$',
's22a2_4': r'$\mathbf{w} \;=\; 0.5\,(2,\,0) \;=\; (1,\,0)\,,\qquad b \;=\; 1-\mathbf{w}^{T}\mathbf{x}_2 \;=\; 1-3 \;=\; -2$',
's22a2_5': r'$\Vert\mathbf{w}\Vert^2 \;=\; 1 \;=\; \alpha_1+\alpha_2\,,\qquad \dfrac{2}{\Vert\mathbf{w}\Vert} \;=\; 2$',
# ---- 22A.3
's22a3_1': [r'$\xi_{A}=\max(0,\,1-1.8)=0\,,\qquad \xi_{B}=\max(0,\,1-1.0)=0$', r'$\xi_{C}=\max(0,\,1-0.4)=0.6\,,\qquad \xi_{D}=\max(0,\,1+0.5)=1.5$'],
's22a3_2': r'$\sum\xi \;=\; 0+0+0.6+1.5 \;=\; 2.1\,,\qquad C\sum\xi \;=\; 2\times2.1 \;=\; 4.2$',
# ---- 22A.4
's22a4_1': r'$\eta \;=\; K_{ii}+K_{jj}-2K_{ij} \;=\; 5+4-2\times2 \;=\; 5$',
's22a4_2': r'$\alpha_j^{\,\mathrm{new}} \;=\; \alpha_j+\dfrac{y_j\,(E_i-E_j)}{\eta} \;=\; 0.1+\dfrac{(-1)(-0.4-0.3)}{5} \;=\; 0.1+0.14 \;=\; 0.24$',
's22a4_3': r'$L \;=\; \max(0,\,\alpha_j-\alpha_i) \;=\; 0\,,\qquad H \;=\; \min(C,\,C+\alpha_j-\alpha_i) \;=\; 1$',
's22a4_4': r'$\alpha_i^{\,\mathrm{new}} \;=\; \alpha_i+y_iy_j\,(\alpha_j^{\,\mathrm{old}}-\alpha_j^{\,\mathrm{new}}) \;=\; 0.1+(-1)(0.1-0.24) \;=\; 0.24$',
's22a4_5': r'$\sum\alpha_iy_i \;=\; 0.1\,(+1)+0.1\,(-1) \;=\; 0 \quad\longrightarrow\quad 0.24\,(+1)+0.24\,(-1) \;=\; 0$',
# ---- 22C
's22c1_1': r'$y_k\,(\mathbf{w}^{T}\mathbf{x}_k+b) \;\geq\; 1 \quad\mathrm{for\ every}\;k\;\mathrm{with}\;\alpha_k=0$',
's22c1_2': r'$y\,f(9,5) \;=\; 0.8\times9+0.4\times5-5 \;=\; 4.2\,,\qquad y\,f(5,2) \;=\; -(0.8\times5+0.4\times2-5) \;=\; 0.2$',
's22c2_1': r'$f(\mathbf{x}) \;=\; \sum_{i}\alpha_iy_i\,e^{-\gamma\Vert\mathbf{x}_i-\mathbf{x}\Vert^2}+b \;\approx\; b \quad\mathrm{when}\;\; \gamma\Vert\mathbf{x}_i-\mathbf{x}\Vert^2\gg1\;\;\mathrm{for\ all}\;i$',
's22c3_1': r'$P(y=1\,\vert\,f) \;=\; \dfrac{1}{1+\exp(A f+B)}\,,\qquad A=-2.26\,,\;\; B=-0.10$',
# ---- questions
's22q1_1': r'$\mathbf{w} \;=\; \sum_{i\in\mathrm{SV}}\alpha_iy_i\,\mathbf{x}_i\,,\qquad \sum_{i\in\mathrm{SV}}\alpha_iy_i \;=\; 0$',
's22q2_1': r'$\alpha_{\mathrm{SGD}} \;=\; \dfrac{1}{C\,n} \quad\Longrightarrow\quad C\,n \;=\; \mathrm{const}\,,\qquad C_{\mathrm{new}} \;=\; \dfrac{C}{10}\;\;\mathrm{for}\;\; n_{\mathrm{new}}=10\,n$',
's22q4_1': r'$\overline{\Vert\mathbf{x}-\mathbf{z}\Vert^2} \;=\; 2\sum_{j=1}^{p}\mathrm{Var}(x_j) \;\;\longrightarrow\;\; 2p \;\;\mathrm{for\ standardised\ features}\,,\qquad \overline{\gamma\,\Vert\mathbf{x}-\mathbf{z}\Vert^2} \;=\; 2$',
}
mkeq(E, '.'); print('solution equations ok:', len(E))
