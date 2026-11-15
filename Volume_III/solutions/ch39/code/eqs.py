# Equation images of the Chapter 39 solutions. Run: python3 code/eqs.py  (writes sol/ch39/s39*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq, matimg
E = {
# ---- 39A.1
's39a1_1': r'$\mathbf{d}=(1,\,2,\,2,\,1)\,,\qquad \mathrm{vol}(V)=1+2+2+1=6$',
's39a1_2': [r'$A=\{1\},\ B=\{2,3,4\}:\qquad \mathrm{cut}=1\,,\quad \mathrm{vol}(A)=1\,,\quad \mathrm{vol}(B)=5$',
            r'$\mathrm{Ncut}=1\left(\dfrac{1}{1}+\dfrac{1}{5}\right)=1.200\,,\qquad \mathrm{RatioCut}=1\left(\dfrac{1}{1}+\dfrac{1}{3}\right)=1.333$'],
's39a1_3': [r'$A=\{1,2\},\ B=\{3,4\}:\qquad \mathrm{cut}=1\,,\quad \mathrm{vol}(A)=3\,,\quad \mathrm{vol}(B)=3$',
            r'$\mathrm{Ncut}=1\left(\dfrac{1}{3}+\dfrac{1}{3}\right)=0.667\,,\qquad \mathrm{RatioCut}=1\left(\dfrac{1}{2}+\dfrac{1}{2}\right)=1.000$'],
's39a1_4': [r'$A=\{1,2,3\},\ B=\{4\}:\qquad \mathrm{cut}=1\,,\quad \mathrm{vol}(A)=5\,,\quad \mathrm{vol}(B)=1$',
            r'$\mathrm{Ncut}=1\left(\dfrac{1}{5}+\dfrac{1}{1}\right)=1.200\,,\qquad \mathrm{RatioCut}=1\left(\dfrac{1}{3}+\dfrac{1}{1}\right)=1.333$'],
# ---- 39A.2
's39a2_1': r'$\mathbf{f}^{T}\mathbf{L}\,\mathbf{f}\;=\;\sum_{\mathrm{edges}}(f_i-f_j)^{2}\;=\;6\cdot 0^{2}+(f_3-f_4)^{2}\;=\;(1-(-1))^{2}\;=\;4$',
's39a2_2': r'$\mathbf{L}\mathbf{f}=(0,\,0,\,2,\,-2,\,0,\,0)^{T}\,,\qquad \mathbf{f}^{T}\mathbf{L}\,\mathbf{f}=1\cdot 2+(-1)(-2)=4$',
's39a2_3': [r'$\mathbf{f}^{T}\mathbf{D}\,\mathbf{f}=\sum_i d_i f_i^{2}=2+2+3+3+2+2=14$',
            r'$\mathbf{f}^{T}\mathbf{D}\,\mathbf{1}=\sum_i d_i f_i=(2+2+3)-(3+2+2)=0$'],
's39a2_4': r'$\dfrac{\mathbf{f}^{T}\mathbf{L}\,\mathbf{f}}{\mathbf{f}^{T}\mathbf{D}\,\mathbf{f}} \;=\; \dfrac{4}{14} \;=\; 0.2857$',
's39a2_5': r'$\mathrm{Ncut}(A,B)=\mathrm{cut}\left(\dfrac{1}{7}+\dfrac{1}{7}\right)=\dfrac{2}{7}=0.2857$',
's39a2_6': r'$\lambda_2=0.2047\ \ \leq\ \ \mathrm{Ncut}=\dfrac{\mathbf{f}^{T}\mathbf{L}\,\mathbf{f}}{\mathbf{f}^{T}\mathbf{D}\,\mathbf{f}}=0.2857$',
# ---- 39A.3
's39a3_2': r'$\mathbf{L}_{\triangle}\mathbf{1}=(3-3)\,\mathbf{1}=\mathbf{0}\,,\qquad \mathbf{L}_{\triangle}\mathbf{v}=3\,\mathbf{v}\ \ \mathrm{when}\ \ \mathbf{1}^{T}\mathbf{v}=0$',
's39a3_3': [r'$\mathbf{1}_A=(1,1,1,0,0,0)^{T}\,,\qquad \mathbf{1}_B=(0,0,0,1,1,1)^{T}$',
            r'$\mathbf{L}\mathbf{1}_A=\mathbf{L}\mathbf{1}_B=\mathbf{0}$'],
's39a3_4': r'$\mathbf{f}^{(1)}=\dfrac{1}{\sqrt{12}}\,(1,1,1,1,1,1)^{T}\,,\qquad \mathbf{f}^{(2)}=\dfrac{1}{\sqrt{12}}\,(1,1,1,-1,-1,-1)^{T}$',
's39a3_5': [r'$\mathbf{u}_i=(0.2887,\ 0.2887)\ \ (i=1,2,3)$',
            r'$\mathbf{u}_i=(0.2887,\ -0.2887)\ \ (i=4,5,6)$'],
# ---- 39A.4
's39a4_1': r'$s(1,2)=-1\,,\quad s(2,3)=-4\,,\quad s(1,3)=-9\,,\quad s(k,k)=p$',
's39a4_2': r'$\mathrm{one:}\ \ -5+p\,,\qquad \mathrm{two:}\ \ -1+2p\,,\qquad \mathrm{three:}\ \ 3p$',
's39a4_3': [r'$-5+p=-1+2p\ \ \Longrightarrow\ \ p=-4$',
            r'$-1+2p=3p\ \ \Longrightarrow\ \ p=-1$'],
# ---- 39C.1, 39C.2
's39c1_1': r'$0\;=\;\mathbf{f}^{T}\mathbf{L}\,\mathbf{f}\;=\;\dfrac{1}{2}\sum_{i,j}w_{ij}\,(f_i-f_j)^{2}\quad\Longrightarrow\quad f_i=f_j\ \ \mathrm{whenever}\ \ w_{ij}>0$',
's39c2_1': r'$\sum_{i}\left[\,s(i,c_i^{\mathrm{new}})-s(i,c_i^{\mathrm{old}})\,\right]\ >\ |p|\,(\mathrm{number\ of\ new\ exemplars})$',
# ---- questions
's39q3_1': r'$\gamma=\dfrac{1}{2\sigma^{2}}\,,\qquad \gamma_{\mathrm{m}^{-2}}=\gamma_{\mathrm{km}^{-2}}\,/\,10^{6}$',
}
mkeq(E, '.')
matimg('s39a3_1', [r'$\mathbf{L}_{\triangle}\;=\;$', [['2','-1','-1'],['-1','2','-1'],['-1','-1','2']], r'$\;=\;3\,\mathbf{I}-\mathbf{1}\mathbf{1}^{T}$'], '.')
print('eqs ok')
