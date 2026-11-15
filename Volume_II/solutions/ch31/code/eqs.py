# Equation images of the Chapter 31 solutions. Run: python3 code/eqs.py  (writes sol/ch31/s31*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq, matimg
W = r'\mathbf{W}^{(1)}'
E = {
# ---- 31A.1
's31a1_1': [r'$z_{11}=2\times0.5+1\times(-1.0)+0.5=0.5$', r'$z_{12}=2\times1.0+1\times0.5+(-0.5)=2.0$'],
's31a1_2': [r'$\mathbf{h}=\max(0,\mathbf{z}_1)=(0.5,\ 2.0)$', r'$z_2=0.5\times1+2.0\times(-1)+0=-1.5$'],
's31a1_3': r'$p=\dfrac{1}{1+e^{-z_2}}=\dfrac{1}{1+e^{1.5}}=\dfrac{1}{1+4.4817}=0.1824$',
's31a1_4': r'$\ell=-\ln(1-p)=-\ln 0.8176=0.2014$',
# ---- 31A.2
's31a2_1': [r'$z_{11}=1\times0.5+2\times(-1.0)+0.5=-1.0\,,\qquad z_{12}=1\times1.0+2\times0.5-0.5=1.5$',
            r'$\mathbf{h}=(0,\ 1.5)\,,\qquad z_2=0\times1+1.5\times(-1)=-1.5\,,\qquad p=0.1824$'],
's31a2_2': [r'$\delta_2=p-y=0.1824-0=0.1824$',
            r'$\boldsymbol{\delta}_1=\delta_2\,(1,\,-1)\odot(0,\,1)=(0,\ -0.1824)$'],
's31a2_3': [r'$\partial J/\partial\mathbf{W}^{(2)}=\mathbf{h}\,\delta_2=(0,\ 1.5\times0.1824)=(0,\ 0.2736)$',
            r'$\partial J/\partial b^{(2)}=\delta_2=0.1824$'],
's31a2_5': [r'$\partial J/\partial\mathbf{b}^{(1)}=\boldsymbol{\delta}_1=(0,\ -0.1824)$'],
# ---- 31A.3
's31a3_1': [r'$(64\times100+100)+(100\times50+50)+(50\times10+10)=6500+5050+510=12060$'],
's31a3_2': r'$e^{2}=7.3891\,,\quad e^{0}=1\,,\quad e^{-1}=0.3679\,,\qquad \sum=8.7569$',
's31a3_3': r'$\mathbf{p}=(0.8438,\ 0.1142,\ 0.0420)\,,\qquad \ell=-\ln p_2=-\ln0.1142=2.1698$',
# ---- 31A.4
's31a4_1': [r'$\mathrm{momentum:}\quad v_t=\mu v_{t-1}-\eta g_t\,,\qquad w_t=w_{t-1}+v_t$',
            r'$\mathrm{Nesterov:}\quad v_t=\mu v_{t-1}-\eta g_t\,,\qquad w_t=w_{t-1}+\mu v_t-\eta g_t$'],
# ---- 31C
's31c1_1': r'$\delta_{1j}=\delta_2\,W^{(2)}_{j}\,\varphi^{\prime}(z_{1j})\,,\qquad z_{1j}=\mathbf{x}\cdot\mathbf{W}^{(1)}_{:,j}+b^{(1)}_{j}$',
's31c1_2': r'$\dfrac{\partial J}{\partial W^{(1)}_{ij}}=\dfrac{1}{n}\sum_{s}x_{si}\,\delta_{1j}^{(s)}$',
's31c2_1': r'$f(x)=b+\sum_{j}v_j\,\max(0,\,w_jx+b_j)\;\longrightarrow\;\mathrm{straight\ line\ beyond\ the\ last\ kink}$',
's31c2_2': r'$f(x)=b+\sum_{j}v_j\tanh(w_jx+b_j)\;\longrightarrow\;b+\sum_{j}v_j\,\mathrm{sign}(w_j)\quad(x\rightarrow\infty)$',
# ---- questions
's31q1_1': r'$\mathbf{A}_3=\left(\left(\mathbf{X}\mathbf{W}_1+\mathbf{1}\mathbf{b}_1^{T}\right)\mathbf{W}_2+\mathbf{1}\mathbf{b}_2^{T}\right)\mathbf{W}_3+\mathbf{1}\mathbf{b}_3^{T}=\mathbf{X}\left(\mathbf{W}_1\mathbf{W}_2\mathbf{W}_3\right)+\mathbf{1}\mathbf{b}^{T}$',
's31q3_1': r'$\dfrac{\partial J}{\partial W^{(1)}_{ij}}=\dfrac{1}{n}\sum_{s}x_{si}\,\delta_{1j}^{(s)}$',
}
mkeq(E, '.')
matimg('s31a2_4', [r'$\partial J/\partial\mathbf{W}^{(1)}=\mathbf{x}^{T}\boldsymbol{\delta}_1=$', [['1'], ['2']], r'$(0,\ -0.1824)=$', [['0', '-0.1824'], ['0', '-0.3649']]], '.')
print('solution equations ok:', len(E) + 1)
