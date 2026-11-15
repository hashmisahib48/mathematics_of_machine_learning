# Equation images of Chapter 37. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq, matimg
E={
'e3731': [r'$d_{\mathrm{single}}(A,B)=\min_{i\in A,\,j\in B}\ d_{ij}\,,\qquad d_{\mathrm{complete}}(A,B)=\max_{i\in A,\,j\in B}\ d_{ij}$',
          r'$d_{\mathrm{average}}(A,B)=\dfrac{1}{|A|\,|B|}\sum_{i\in A}\ \sum_{j\in B}\ d_{ij}$'],
'e3732': r'$\mathrm{SS}(C)\;=\;\sum_{i\in C}\,\|\mathbf{x}_i-\boldsymbol{\mu}_C\|^{2}\,,\qquad \boldsymbol{\mu}_C\;=\;\dfrac{1}{|C|}\sum_{i\in C}\mathbf{x}_i$',
'e3733': r'$\Delta(A,B)\;=\;\mathrm{SS}(A\cup B)\;-\;\mathrm{SS}(A)\;-\;\mathrm{SS}(B)$',
'e3734': r'$\mathrm{SS}(A\cup B)\;=\;\mathrm{SS}(A)\;+\;\mathrm{SS}(B)\;+\;n_A\,\|\boldsymbol{\mu}_A-\boldsymbol{\mu}\|^{2}\;+\;n_B\,\|\boldsymbol{\mu}_B-\boldsymbol{\mu}\|^{2}$',
'e3735': r'$\boldsymbol{\mu}_A-\boldsymbol{\mu}\;=\;\dfrac{n_B}{n_A+n_B}\,(\boldsymbol{\mu}_A-\boldsymbol{\mu}_B)\,,\qquad \boldsymbol{\mu}_B-\boldsymbol{\mu}\;=\;-\dfrac{n_A}{n_A+n_B}\,(\boldsymbol{\mu}_A-\boldsymbol{\mu}_B)$',
'e3736': r'$\Delta(A,B)\;=\;\dfrac{n_A\,n_B}{n_A+n_B}\;\|\boldsymbol{\mu}_A-\boldsymbol{\mu}_B\|^{2}$',
'e3737': r'$h(A,B)\;=\;\sqrt{2\,\Delta(A,B)}\;=\;\sqrt{\dfrac{2\,n_A\,n_B}{n_A+n_B}}\;\|\boldsymbol{\mu}_A-\boldsymbol{\mu}_B\|$',
'e3738': r'$d(C,\,A\cup B)\;=\;\alpha_A\,d(C,A)\;+\;\alpha_B\,d(C,B)\;+\;\beta\,d(A,B)\;+\;\gamma\,\left|\,d(C,A)-d(C,B)\,\right|$',
'e3741': r'$K_{\mathrm{out}}\;=\;1\;+\;\#\{\,k:\ h_k\ \geq\ t\,\}$',
'e3751': r'$\mathrm{CF}\;=\;(N,\ \mathbf{LS},\ SS)\,,\qquad N=\left|C\right|,\quad \mathbf{LS}=\sum_{i\in C}\mathbf{x}_i\,,\quad SS=\sum_{i\in C}\|\mathbf{x}_i\|^{2}$',
'e3752': r'$\mathrm{CF}_1+\mathrm{CF}_2\;=\;\left(N_1+N_2,\ \ \mathbf{LS}_1+\mathbf{LS}_2,\ \ SS_1+SS_2\right)$',
'e3753': r'$\mathbf{c}\;=\;\dfrac{\mathbf{LS}}{N}\,,\qquad R\;=\;\sqrt{\dfrac{1}{N}\sum_{i\in C}\|\mathbf{x}_i-\mathbf{c}\|^{2}}\;=\;\sqrt{\dfrac{SS}{N}-\|\mathbf{c}\|^{2}}$',
'e3711': r'$d(C,A\cup B)\;=\;\dfrac{(n_A+n_C)\,d_{CA}+(n_B+n_C)\,d_{CB}-n_C\,d_{AB}}{n_A+n_B+n_C}\;\geq\;d_{AB}$',
# worked examples
'e3761': r'$\Delta\;=\;\dfrac{2}{3}\,\|\mathbf{x}_A-\boldsymbol{\mu}_{BC}\|^{2}\;=\;\dfrac{2}{3}\left[(3-0.5)^{2}+(3-2)^{2}\right]\;=\;\dfrac{2}{3}\times 7.25\;=\;4.8333$',
'e3762': r'$d^{2}\;=\;\dfrac{2}{3}\,(5)+\dfrac{2}{3}\,(10)-\dfrac{1}{3}\,(1)\;=\;\dfrac{29}{3}\;=\;9.6667\,,\qquad \Delta\;=\;\dfrac{9.6667}{2}\;=\;4.8333$',
}
mkeq(E,'.')
print('eqs ok')
