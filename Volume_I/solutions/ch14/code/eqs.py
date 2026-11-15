# Equation images of the Chapter 14 solutions. Run: python3 code/eqs.py  (writes sol/ch14/s14*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq, matimg
E = {
# ---- 14A.1
's14a1_1': r'$\ell(\mu) \;=\; \sum_i\,(y_i\ln\mu-\mu) \;=\; 5\ln\mu-3\mu$',
's14a1_2': r'$\dfrac{d\ell}{d\mu} \;=\; \dfrac{5}{\mu}-3 \;=\; 0 \quad\Longrightarrow\quad \hat{\mu} \;=\; \dfrac{5}{3} \;=\; 1.6667$',
's14a1_3': [r'$d(2) \;=\; 2\,[\,2\ln 1.2-0.3333\,] \;=\; 0.0626$',
            r'$d(0) \;=\; 2\hat{\mu} \;=\; 3.3333$',
            r'$d(3) \;=\; 2\,[\,3\ln 1.8-1.3333\,] \;=\; 0.8601$'],
's14a1_4': r'$D \;=\; 0.0626+3.3333+0.8601 \;=\; 4.2560$',
's14a1_5': r'$2\left[\ell_{\mathrm{sat}}-\ell(\hat{\mu})\right] \;=\; 2\,[\,-0.3179-(-2.4459)\,] \;=\; 4.2560$',
# ---- 14A.2
's14a2_1': [r'$\mu=2:\qquad d_{\mathrm{P}} \;=\; 2\,[\,3\ln 1.5-1\,] \;=\; 0.4328$',
            r'$\mu=4.5:\qquad d_{\mathrm{P}} \;=\; 2\,[\,3\ln(2/3)+1.5\,] \;=\; 0.5672$'],
's14a2_2': [r'$\mu=2:\qquad d_{\mathrm{G}} \;=\; 2\,[\,0.5-\ln 1.5\,] \;=\; 0.1891$',
            r'$\mu=4.5:\qquad d_{\mathrm{G}} \;=\; 2\,[\,-0.3333-\ln(2/3)\,] \;=\; 0.1443$'],
# ---- 14A.3
's14a3_1': [r'$\mathbf{y}-\boldsymbol{\mu} \;=\; (-2,\,0,\,2)$',
            r'$\mathbf{g} \;=\; \left(\sum(y-\mu),\ \sum x\,(y-\mu)\right) \;=\; (0,\ 0+0+4) \;=\; (0,\,4)$'],
# ---- questions
's14q1_1': [r'$E\left[\ln y\right] \;=\; \ln\mu+\psi(k)-\ln k \;<\; \ln\mu$',
            r'$e^{\psi(k)}/k \;=\; 0.624 \quad(k=1.2)$'],
's14q3_1': r'$d_{\mathrm{P}}(N,\ e\lambda) \;=\; e\; d_{\mathrm{P}}\!\left(\dfrac{N}{e},\ \lambda\right)$',
's14q5_1': r'$d_{p}(c\,y,\ c\,\mu) \;=\; c^{\,2-p}\; d_{p}(y,\ \mu)$',
}
mkeq(E, '.')
matimg('s14a3_2', [r'$\mathbf{H} \;=\; 3\,\mathbf{X}^{T}\mathbf{X} \;=\; 3$', [['3', '3'], ['3', '5']], r'$=$', [['9', '9'], ['9', '15']]], '.', rh=0.28)
matimg('s14a3_3', [r'$\boldsymbol{\delta} \;=\; \mathbf{H}^{-1}\mathbf{g} \;=\; \dfrac{1}{54}$', [['15', '-9'], ['-9', '9']], [['0'], ['4']], r'$=\ \dfrac{1}{54}$', [['-36'], ['36']], r'$=$', [['-0.6667'], ['0.6667']]], '.', rh=0.28)
print('solution equations ok:', len(E) + 2)
