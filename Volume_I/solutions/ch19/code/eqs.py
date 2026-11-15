# Equation images of the Chapter 19 solutions. Run: python3 code/eqs.py  (writes sol/ch19/s19*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
from PIL import Image
E = {
# ---- 19A.1
's19a1_1': [r'$\theta_{\mathrm{ham},\,\mathrm{lunch}} \;=\; \dfrac{2+1}{8+7} \;=\; \dfrac{3}{15}\,, \qquad \theta_{\mathrm{ham},\,\mathrm{now}} \;=\; \dfrac{1+1}{15} \;=\; \dfrac{2}{15}$',
             r'$\theta_{\mathrm{spam},\,\mathrm{lunch}} \;=\; \dfrac{0+1}{10+7} \;=\; \dfrac{1}{17}\,, \qquad \theta_{\mathrm{spam},\,\mathrm{now}} \;=\; \dfrac{3+1}{17} \;=\; \dfrac{4}{17}$'],
's19a1_2': [r'$\mathrm{ham:}\quad 0.5\times\dfrac{3}{15}\times\left(\dfrac{2}{15}\right)^{2} \;=\; 0.001778$',
             r'$\mathrm{spam:}\quad 0.5\times\dfrac{1}{17}\times\left(\dfrac{4}{17}\right)^{2} \;=\; 0.001628$'],
's19a1_3': r'$P(\mathrm{spam}\,|\,\mathbf{x}) \;=\; \dfrac{0.001628}{0.001778+0.001628} \;=\; 0.4781$',
's19a1_4': [r'$\ln\dfrac{\theta_{\mathrm{spam},\mathrm{lunch}}}{\theta_{\mathrm{ham},\mathrm{lunch}}} + 2\ln\dfrac{\theta_{\mathrm{spam},\mathrm{now}}}{\theta_{\mathrm{ham},\mathrm{now}}}$',
             r'$\;=\; -1.2238 + 2\times0.5680 \;=\; -0.0878$'],
# ---- 19A.2
's19a2_1': [r'$\boldsymbol{\mu}_0=(2,\,3)\,,\quad \sigma_0^{2}=\dfrac{1+0+1}{3}=0.6667\ \ \mathrm{for\ both\ features}$',
             r'$\boldsymbol{\mu}_1=(6,\,7)\,,\quad \sigma_1^{2}=\dfrac{4+0+4}{3}=2.6667\ \ \mathrm{for\ both\ features}$'],
's19a2_2': [r'$s_0 \;=\; \ln 0.5 - \dfrac{1}{2}\left[2\ln(2\pi\times0.6667) + \dfrac{1.5^2+1^2}{0.6667}\right]$',
             r'$\;=\; -0.6931 - \dfrac{1}{2}\left[2.8648 + 4.8750\right] \;=\; -4.5631$'],
's19a2_3': [r'$s_1 \;=\; \ln 0.5 - \dfrac{1}{2}\left[2\ln(2\pi\times2.6667) + \dfrac{2.5^2+3^2}{2.6667}\right]$',
             r'$\;=\; -0.6931 - \dfrac{1}{2}\left[5.6374 + 5.7188\right] \;=\; -6.3712$'],
's19a2_4': r'$P(0\,|\,\mathbf{x}) \;=\; \dfrac{1}{1+e^{\,s_1-s_0}} \;=\; \dfrac{1}{1+e^{-1.8082}} \;=\; 0.8591$',
# ---- 19A.3
's19a3_1': r'$\theta_{kj} \;=\; \dfrac{N_{kj}+1}{n_k+2} \;=\; \dfrac{N_{kj}+1}{5}\,, \qquad n_k=3$',
's19a3_2': [r'$s_{\mathrm{spam}} \;=\; \ln0.5 + \left[\ln0.8+\ln0.2+\ln0.6\right]$',
             r'$\qquad\quad +\ \left[\ln0.8+\ln0.4+\ln0.8+\ln0.4\right]$',
             r'$\;=\; -0.6931 - 2.3434 - 2.2789 \;=\; -5.3154$'],
's19a3_2b': [r'$s_{\mathrm{ham}} \;=\; \ln0.5 + \left[\ln0.2+\ln0.8+\ln0.4\right]$',
             r'$\qquad\quad +\ \left[\ln0.4+\ln0.8+\ln0.4+\ln0.8\right]$',
             r'$\;=\; -0.6931 - 2.7489 - 2.2789 \;=\; -5.7209$'],
's19a3_3': r'$P(\mathrm{spam}\,|\,\mathbf{x}) \;=\; \dfrac{1}{1+e^{-5.7209+5.3154}} \;=\; \dfrac{1}{1+e^{-0.4055}} \;=\; 0.6000$',
# ---- 19A.4
's19a4_1': r'$\theta_c \;=\; \dfrac{N_c+1}{4+m}\,, \qquad N=(1,\,2,\,1)$',
's19a4_2': [r'$m=3:\quad \theta=\left(\dfrac{2}{7},\ \dfrac{3}{7},\ \dfrac{2}{7}\right)=(0.2857,\ 0.4286,\ 0.2857)$',
             r'$m=4:\quad \theta=\left(\dfrac{2}{8},\ \dfrac{3}{8},\ \dfrac{2}{8},\ \dfrac{1}{8}\right)=(0.25,\ 0.375,\ 0.25,\ 0.125)$'],
# ---- 19B.1
's19b1_1': r'$\dfrac{P(1\,|\,\mathbf{x})}{P(0\,|\,\mathbf{x})} \;=\; \dfrac{P(1)}{P(0)} \times \exp\left(\sum_j x_j\ln\dfrac{\theta_{1j}}{\theta_{0j}}\right) \;=\; \dfrac{P(1)}{P(0)}\times1.3739$',
# ---- 19B.2
's19b2_1': r'$\theta_{kj} \;=\; \dfrac{N_{kj}+\alpha}{N_k+\alpha p} \;\longrightarrow\; \dfrac{\alpha}{\alpha p} \;=\; \dfrac{1}{p} \qquad (\alpha\to\infty)$',
# ---- 19C.1
's19c1_1': r'$s_k^{\,(r)} \;=\; \ln P(k) + r\,\ln p(x_1\,|\,k) + \sum_{j\geq2}\ln p(x_j\,|\,k)$',
's19c1_2': r'$\ln\dfrac{P(1\,|\,\mathbf{x})}{P(0\,|\,\mathbf{x})} \;=\; 4.2387 - 1.6306\,(r-1)$',
# ---- questions
's19q1_1': r'$p(\mathbf{x}\,|\,k) \;=\; \prod_{j=1}^{p} p(x_j\,|\,k) \qquad \mathrm{but}\qquad p(\mathbf{x}) \;\neq\; \prod_{j} p(x_j)$',
's19q3_1': r'$\mathrm{posterior\ odds} \;=\; \dfrac{0.001}{0.999}\times\mathrm{likelihood\ ratio}$',
}
mkeq(E, '.'); print('solution equations ok:', len(E))
for k in E:
    with Image.open(k + '.png') as im:
        if im.size[0] > 1250: print('  too wide (will be shrunk):', k, im.size[0], 'px')
