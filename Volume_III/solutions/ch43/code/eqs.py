# Equation images of the Chapter 43 solutions. Run: python3 code/eqs.py  (writes sol/ch43/s43*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 43A.1
's43a1_1': r'$(\mathbf{W}\mathbf{H}_0)_{1j}=1\cdot1+2\cdot1=3\,,\qquad(\mathbf{W}\mathbf{H}_0)_{2j}=2\cdot1+1\cdot1=3$',
's43a1_2': r'$d_1(3\,|\,3)=0\,,\qquad d_1(6\,|\,3)=6\ln\dfrac{6}{3}-6+3=6\ln2-3=1.1589$',
's43a1_3': r'$D_1(\mathbf{X}\,|\,\mathbf{W}\mathbf{H}_0)\;=\;2\times0+2\times1.1589\;=\;2.3178$',
's43a1_4': r'$H_{11}\;\leftarrow\;H_{11}\,\dfrac{W_{11}\,x_{11}/y_{11}+W_{21}\,x_{21}/y_{21}}{W_{11}+W_{21}}\;=\;1\cdot\dfrac{1\cdot1+2\cdot2}{1+2}\;=\;\dfrac{5}{3}$',
's43a1_5': r'$(\mathbf{W}\mathbf{H}_1)_{11}=1\cdot\dfrac{5}{3}+2\cdot\dfrac{4}{3}=\dfrac{13}{3}\,,\qquad(\mathbf{W}\mathbf{H}_1)_{12}=1\cdot\dfrac{4}{3}+2\cdot\dfrac{5}{3}=\dfrac{14}{3}$',
's43a1_6': r'$d_1\left(3\,|\,\dfrac{13}{3}\right)=3\ln\dfrac{9}{13}-3+\dfrac{13}{3}=0.2302$',
's43a1_9': r'$d_1\left(6\,|\,\dfrac{14}{3}\right)=6\ln\dfrac{9}{7}-6+\dfrac{14}{3}=0.1746$',
's43a1_7': r'$D_1(\mathbf{X}\,|\,\mathbf{W}\mathbf{H}_1)\;=\;2\times0.2302+2\times0.1746\;=\;0.8094$',
's43a1_8': r'$\sum_{ij}(\mathbf{W}\mathbf{H}_1)_{ij}=2\times\dfrac{13}{3}+2\times\dfrac{14}{3}=18=\sum_{ij}x_{ij}$',
# ---- 43A.2
's43a2_1': r'$d_2(4\,|\,2)=\frac{1}{2}(4-2)^2=2\,,\qquad d_2(40\,|\,20)=\frac{1}{2}(40-20)^2=200$',
's43a2_2': r'$d_1(4\,|\,2)=4\ln2-4+2=0.7726\,,\quad d_1(40\,|\,20)=40\ln2-20=7.7259$',
's43a2_3': r'$\dfrac{200}{2}=100=10^{2}\,,\qquad\dfrac{7.7259}{0.7726}=10=10^{1}$',
's43a2_4': r'$d_0(4\,|\,2)=\dfrac{4}{2}-\ln\dfrac{4}{2}-1=1-\ln2=0.3069=d_0(40\,|\,20)$',
# ---- 43A.3
's43a3_1': r'$\kappa\;=\;\dfrac{E[y^{4}]}{E[y^{2}]^{2}}-3\,,\qquad E[\,\cdot\,]=\mathrm{average\ over\ the\ samples}$',
's43a3_2': r'$\mathrm{a:}\quad E[y^2]=\dfrac{6\times1}{6}=1\,,\quad E[y^4]=\dfrac{6\times1}{6}=1\,,\quad\kappa=\dfrac{1}{1^2}-3=-2$',
's43a3_3': r'$\mathrm{b:}\quad E[y^2]=\dfrac{4+1+0+0+1+4}{6}=1.6667$',
's43a3_6': r'$E[y^4]=\dfrac{16+1+0+0+1+16}{6}=5.6667\,,\quad\kappa=\dfrac{5.6667}{1.6667^{2}}-3=-0.96$',
's43a3_5': r'$\mathrm{c:}\quad E[y^2]=\dfrac{9+9}{10}=1.8\,,\quad E[y^4]=\dfrac{81+81}{10}=16.2$',
's43a3_7': r'$\kappa=\dfrac{16.2}{1.8^{2}}-3=5-3=2$',
# ---- 43C.1
's43c1_1': r'$\mathbf{X}\;=\;\mathbf{X}\,\mathbf{I}\,,\qquad\mathbf{W}=\mathbf{X}\;(\mathrm{rows}\ (4,2),\,(3,3),\,(2,4))\,,\quad\mathbf{H}=\mathbf{I}$',
's43c1_2': r'$\mathbf{W}\mathbf{H}\;=\;(\mathbf{W}\mathbf{D}^{-1}\mathbf{P}^{T})\,(\mathbf{P}\mathbf{D}\mathbf{H})\,,\qquad\mathbf{D}=\mathrm{diag}(d_1,d_2)>0\,,\quad\mathbf{P}\ \mathrm{a\ permutation}$',
's43c1_3': r'$\mathbf{W}\;=\;\mathbf{X}\mathbf{H}^{-1}\;\geq\;0\quad\Longleftrightarrow\quad0^\circ\leq\theta_1\leq26.57^\circ\,,\quad63.43^\circ\leq\theta_2\leq90^\circ$',
# ---- 43C.2
's43c2_1': r'$p(\mathbf{s})\;=\;\dfrac{1}{2\pi}\,e^{-(s_1^2+s_2^2)/2}\;=\;\dfrac{1}{2\pi}\,e^{-\Vert\mathbf{s}\Vert^{2}/2}\,,\qquad\Vert\mathbf{R}\mathbf{s}\Vert=\Vert\mathbf{s}\Vert$',
's43c2_2': r'$\mathbf{x}\;=\;\mathbf{A}\mathbf{s}\;=\;(\mathbf{A}\mathbf{R}^{T})(\mathbf{R}\mathbf{s})\;=\;\mathbf{A}^{\prime}\mathbf{s}^{\prime}\,,\qquad\mathbf{s}^{\prime}\sim\mathbf{s}$',
# ---- questions
's43q3_1': r'$\boldsymbol{\Sigma}\;=\;\mathbf{L}\mathbf{L}^{T}+\sigma^{2}\mathbf{I}\quad\Longrightarrow\quad\boldsymbol{\Sigma}\mathbf{v}=\lambda\mathbf{v}\ \Longleftrightarrow\ \mathbf{L}\mathbf{L}^{T}\mathbf{v}=(\lambda-\sigma^{2})\mathbf{v}$',
}
mkeq(E, '.')
from PIL import Image
for k in E:
    w = Image.open(k + '.png').size[0]
    if w > 1300: print('too wide', k, w)
