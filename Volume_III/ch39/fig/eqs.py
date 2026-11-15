# Equation and matrix images of Chapter 39. Run: python3 fig/eqs.py
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq, matimg
E = {
# 39.3 spectral clustering
'e3931': r'$w_{ij} \;=\; \exp\left(-\gamma\,\Vert \mathbf{x}_i-\mathbf{x}_j \Vert^{2}\right) \,, \qquad \gamma=\dfrac{1}{2\sigma^{2}}$',
'e3932': r'$w_{ij} \;=\; \dfrac{1}{2}\left(c_{ij}+c_{ji}\right)$',
'e3933': [r'$d_i \;=\; \sum_{j} w_{ij}\,, \qquad \mathbf{D}=\mathrm{diag}(d_1,\dots,d_n)$',
          r'$\mathbf{L} \;=\; \mathbf{D}-\mathbf{W}$'],
'e3934': [r'$\mathbf{f}^{T}\mathbf{L}\,\mathbf{f} \;=\; \sum_i\, d_i f_i^{2} \;-\; \sum_{i,j}\, w_{ij}\, f_i f_j$',
          r'$=\; \dfrac{1}{2}\sum_{i,j}\, w_{ij}\,\left(f_i^{2}+f_j^{2}-2f_if_j\right) \;=\; \dfrac{1}{2}\sum_{i,j}\, w_{ij}\,\left(f_i-f_j\right)^{2}$'],
'e3935': [r'$\mathrm{cut}(A,B) \;=\; \sum_{i\in A}\sum_{j\in B} w_{ij}\,, \qquad \mathrm{vol}(A) \;=\; \sum_{i\in A} d_i$',
          r'$\mathrm{Ncut}(A,B) \;=\; \mathrm{cut}(A,B)\left(\dfrac{1}{\mathrm{vol}(A)}+\dfrac{1}{\mathrm{vol}(B)}\right)$'],
'e3936': r'$f_i \;=\; \sqrt{v_B/v_A}\ \ (i\in A)\,, \qquad f_i \;=\; -\sqrt{v_A/v_B}\ \ (i\in B)$',
'e3937': [r'$\mathbf{f}^{T}\mathbf{L}\,\mathbf{f} \;=\; \mathrm{cut}(A,B)\left(\sqrt{v_B/v_A}+\sqrt{v_A/v_B}\right)^{2}$',
          r'$=\; \mathrm{cut}(A,B)\,(v_A+v_B)^{2}/(v_Av_B) \;=\; \mathrm{vol}(V)\,\mathrm{Ncut}(A,B)$',
          r'$\mathbf{f}^{T}\mathbf{D}\,\mathbf{f} \;=\; v_A\,(v_B/v_A)+v_B\,(v_A/v_B) \;=\; \mathrm{vol}(V)$',
          r'$\mathbf{f}^{T}\mathbf{D}\,\mathbf{1} \;=\; v_A\sqrt{v_B/v_A}-v_B\sqrt{v_A/v_B} \;=\; 0$'],
'e3938': r'$\min_{\mathbf{f}\in\mathbb{R}^{n}}\ \dfrac{\mathbf{f}^{T}\mathbf{L}\,\mathbf{f}}{\mathbf{f}^{T}\mathbf{D}\,\mathbf{f}} \quad \mathrm{subject\ to}\quad \mathbf{f}^{T}\mathbf{D}\,\mathbf{1}=0 \qquad\Longrightarrow\qquad \mathbf{L}\,\mathbf{f}=\lambda\,\mathbf{D}\,\mathbf{f}$',
'e3939': [r'$\mathbf{g}=\mathbf{D}^{1/2}\mathbf{f}\,: \qquad \mathbf{L}_{\mathrm{sym}}\,\mathbf{g} \;=\; \lambda\,\mathbf{g}$',
          r'$\mathbf{L}_{\mathrm{sym}} \;=\; \mathbf{D}^{-1/2}\mathbf{L}\,\mathbf{D}^{-1/2} \;=\; \mathbf{I}-\mathbf{D}^{-1/2}\mathbf{W}\,\mathbf{D}^{-1/2}$'],
'e39310': r'$\lambda_2 \;=\; \min_{\mathbf{g}\perp\mathbf{D}^{1/2}\mathbf{1}}\ \dfrac{\mathbf{g}^{T}\mathbf{L}_{\mathrm{sym}}\,\mathbf{g}}{\mathbf{g}^{T}\mathbf{g}} \;\leq\; \min_{A,B}\ \mathrm{Ncut}(A,B)$',
'e39311': [r'$\mathbf{L}_{\mathrm{sym}}\mathbf{g}^{(m)}=\lambda_m\mathbf{g}^{(m)}\,, \quad \lambda_1\leq\cdots\leq\lambda_K\,, \qquad \mathbf{f}^{(m)}=\mathbf{D}^{-1/2}\mathbf{g}^{(m)}$',
          r'$\mathbf{u}_{i} \;=\; \left(f^{(1)}_i,\ \dots,\ f^{(K)}_i\right)\,, \qquad i=1,\dots,n$'],
# 39.4 affinity propagation
'e3941': r'$s(i,k) \;=\; -\Vert\mathbf{x}_i-\mathbf{x}_k\Vert^{2}\ \ (i\neq k)\,, \qquad s(k,k) \;=\; p\ \ (\mathrm{the\ preference})$',
'e3942': r'$r(i,k) \;\leftarrow\; s(i,k)\;-\;\max_{k^{\prime}\neq k}\left\{\,a(i,k^{\prime})+s(i,k^{\prime})\,\right\}$',
'e3943': r'$a(i,k) \;\leftarrow\; \min\left\{\,0\,,\ r(k,k)+\sum_{i^{\prime}\notin\{i,k\}}\max\left\{0,\,r(i^{\prime},k)\right\}\right\} \qquad (i\neq k)$',
'e3944': r'$a(k,k) \;\leftarrow\; \sum_{i^{\prime}\neq k}\max\left\{0,\,r(i^{\prime},k)\right\}$',
'e3945': r'$m_{\mathrm{new}} \;=\; \lambda\,m_{\mathrm{old}}+(1-\lambda)\,m_{\mathrm{computed}} \,, \qquad 0.5\leq\lambda<1$',
'e3946': r'$\max_{c}\ \sum_{i=1}^{n} s(i,c_i) \,, \qquad c_i\in\{1,\dots,n\}\,, \quad c_k=k\ \ \mathrm{whenever\ some}\ c_i=k$',
# 39.5 biclustering
'e3952': r'$\mathbf{A}_n \;=\; \mathbf{D}_1^{-1/2}\,\mathbf{A}\,\mathbf{D}_2^{-1/2}$',
'e3953': [r'$\mathbf{A}\,\mathbf{y} \;=\; (1-\lambda)\,\mathbf{D}_1\,\mathbf{x}\,, \qquad \mathbf{A}^{T}\mathbf{x} \;=\; (1-\lambda)\,\mathbf{D}_2\,\mathbf{y}$',
          r'$\mathbf{u}=\mathbf{D}_1^{1/2}\mathbf{x},\ \ \mathbf{v}=\mathbf{D}_2^{1/2}\mathbf{y}\ :\qquad \mathbf{A}_n\mathbf{v}=\sigma\,\mathbf{u}\,, \quad \mathbf{A}_n^{T}\mathbf{u}=\sigma\,\mathbf{v}\,, \quad \sigma=1-\lambda$'],
'e3955': r'$\tilde a_{ij} \;=\; \log a_{ij}-\overline{\log a}_{i\cdot}-\overline{\log a}_{\cdot j}+\overline{\log a}_{\cdot\cdot}$',
'e3956': r'$J(a,b) \;=\; \dfrac{\left|I_a\cap I_b\right|\,\left|J_a\cap J_b\right|}{\left|I_a\right|\left|J_a\right|+\left|I_b\right|\left|J_b\right|-\left|I_a\cap I_b\right|\left|J_a\cap J_b\right|}$',
}
mkeq(E, '.')
# 39.6 by hand: the Laplacian of the six-point graph
matimg('e3961', [r'$\mathbf{L} \;=\;$', [['2','-1','-1','0','0','0'],['-1','2','-1','0','0','0'],['-1','-1','3','-1','0','0'],['0','0','-1','3','-1','-1'],['0','0','0','-1','2','-1'],['0','0','0','-1','-1','2']], r'$,\qquad \mathbf{D}=\mathrm{diag}(2,2,3,3,2,2)$'], '.', rh=0.21)
matimg('e3951', [r'$\mathbf{M} \;=\;$', [[r'\mathbf{0}', r'\mathbf{A}'], [r'\mathbf{A}^{T}', r'\mathbf{0}']], r'$,\qquad \mathbf{D}_1=\mathrm{diag}\left(\mathbf{A}\mathbf{1}\right)\,,\quad \mathbf{D}_2=\mathrm{diag}\left(\mathbf{A}^{T}\mathbf{1}\right)$'], '.', rh=0.24)
matimg('e3954', [r'$\mathbf{Z} \;=\;$', [[r'\mathbf{D}_1^{-1/2}\,\mathbf{U}'], [r'\mathbf{D}_2^{-1/2}\,\mathbf{V}']], r'$,\qquad \left((m+n)\times\ell\right)\,,\quad \ell=\lceil\log_2 K\rceil$'], '.', rh=0.28)
print('eqs ok')
