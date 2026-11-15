# Equation and matrix images of Chapter 38. Run: python3 fig/eqs.py
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq, matimg
E = {
'e3831': r'$N_\varepsilon(i) \;=\; \{\, j \;:\; \Vert \mathbf{x}_i-\mathbf{x}_j \Vert \leq \varepsilon \,\}$',
'e3832': r'$i \ \mathrm{is\ a\ core\ point} \quad\Longleftrightarrow\quad \left|\,N_\varepsilon(i)\,\right| \;\geq\; m$',
'e3833': r'$c_m(i) \;=\; m\mathrm{th\ smallest\ of}\ \left\{\, \Vert \mathbf{x}_i-\mathbf{x}_j \Vert \,:\, j=1,\dots,n \,\right\}$',
'e3834': r'$i=p_0\ \rightarrow\ p_1\ \rightarrow\ \cdots\ \rightarrow\ p_r=j$',
'e3841': r'$d_{\mathrm{mr}}(i,j) \;=\; \max\left\{\, c_m(i)\,,\ c_m(j)\,,\ \Vert \mathbf{x}_i-\mathbf{x}_j \Vert \,\right\}$',
'e3842': r'$\lambda \;=\; \dfrac{1}{\varepsilon}$',
'e3843': r'$S(C) \;=\; \sum_{i\in C}\,\left(\lambda_i-\lambda_{\mathrm{birth}}(C)\right)$',
'e3844': r'$\tilde S(C) \;=\; \max\left\{\, S(C)\,,\ \sum_{j}\tilde S(C_j) \,\right\}$',
'e3845': r'$q_i \;=\; \dfrac{\min\left\{\lambda_i\,,\ \lambda_{\max}(C)\right\}}{\lambda_{\max}(C)}$',
'e3851': [r'$r_p(o) \;=\; \max\left\{\, c_m(p)\,,\ \Vert \mathbf{x}_p-\mathbf{x}_o \Vert \,\right\}$',
          r'$\mathrm{reach}(o) \;=\; \min\left\{\, r_p(o)\ :\ p\ \mathrm{already\ processed\ and\ core} \,\right\}$'],
'e3861': r'$\hat{f}(\mathbf{x}) \;=\; \dfrac{1}{n\,h^{p}}\sum_{i=1}^{n} K\!\left(\dfrac{\mathbf{x}-\mathbf{x}_i}{h}\right)\,, \qquad K(\mathbf{u})=c_k\,k\left(\Vert\mathbf{u}\Vert^{2}\right)$',
'e3862': r'$\nabla\hat{f}(\mathbf{x}) \;=\; \dfrac{2c_k}{n\,h^{p+2}}\sum_{i=1}^{n}\left(\mathbf{x}_i-\mathbf{x}\right) g_i\,, \qquad g_i = g\left(\Vert(\mathbf{x}-\mathbf{x}_i)/h\Vert^{2}\right)$',
'e3863': r'$\nabla\hat{f}(\mathbf{x}) \;=\; \dfrac{2c_k}{n\,h^{p+2}}\left(\sum_{i} g_i\right)\left[\,\dfrac{\sum_{i} g_i\,\mathbf{x}_i}{\sum_{i} g_i}-\mathbf{x}\right]$',
'e3864': [r'$\mathbf{x}\ \leftarrow\ \left(\sum_{i} g_i\,\mathbf{x}_i\right)/\left(\sum_{i} g_i\right)$',
          r'$g_i=\mathbf{1}\left[\Vert\mathbf{x}-\mathbf{x}_i\Vert\leq h\right]\ \ \mathrm{(flat)}\,, \qquad g_i=e^{-\Vert\mathbf{x}-\mathbf{x}_i\Vert^{2}/2h^{2}}\ \ \mathrm{(Gaussian)}$'],
'e3865': r'$h \;=\; \dfrac{1}{n}\sum_{i=1}^{n} d_{(k)}(i)\,, \qquad k=\lfloor q\,n\rfloor$',
}
matimg('e3871', [r'$\mathbf{x}^{(1)} \;=\; \dfrac{1}{3}\left[\right.$', [['2'],['1']], r'$+$', [['2'],['2']], r'$+$', [['3'],['2']], r'$\left.\right] \;=\;$', [['7/3'],['5/3']], r'$\approx$', [['2.333'],['1.667']]], '.')
matimg('e3872', [r'$\mathbf{x}^{(2)} \;=\; \dfrac{1}{5}\left[\right.$', [['1'],['1']], r'$+$', [['2'],['1']], r'$+$', [['2'],['2']], r'$+$', [['3'],['2']], r'$+$', [['1'],['2']], r'$\left.\right] \;=\;$', [['9/5'],['8/5']], r'$=$', [['1.8'],['1.6']]], '.')
mkeq(E, '.')
print('eqs ok')
