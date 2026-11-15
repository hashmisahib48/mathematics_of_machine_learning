# Equation images of the Chapter 18 solutions. Run: python3 code/eqs.py  (writes sol/ch18/s18*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
from PIL import Image
E = {
# ---- 18A.1
's18a1_1': r'$w \;=\; \dfrac{\mu_1-\mu_0}{\sigma^2} \;=\; \dfrac{6-2}{2} \;=\; 2$',
's18a1_2': [r'$x^{*} \;=\; \dfrac{\mu_0+\mu_1}{2} + \dfrac{\sigma^2}{\mu_1-\mu_0}\,\ln\dfrac{\pi_0}{\pi_1}$',
            r'$\;=\; 4 + \dfrac{2}{4}\,\ln\dfrac{0.7}{0.3} \;=\; 4 + 0.5\times0.8473 \;=\; 4.4236$'],
's18a1_3': [r'$b \;=\; -\dfrac{\mu_1^2-\mu_0^2}{2\sigma^2} + \ln\dfrac{\pi_1}{\pi_0}$',
            r'$\;=\; -\dfrac{36-4}{4} + \ln\dfrac{0.3}{0.7} \;=\; -8-0.8473 \;=\; -8.8473$'],
's18a1_4': [r'$\ln\dfrac{P(1\,|\,4.5)}{P(0\,|\,4.5)} \;=\; 2\times4.5 - 8.8473 \;=\; 0.1527$',
            r'$P(1\,|\,4.5) \;=\; \dfrac{1}{1+e^{-0.1527}} \;=\; 0.5381$'],
# ---- 18A.2
's18a2_1': r'$\Sigma^{-1} \;=\; \mathrm{diag}\left(\dfrac{1}{2},\,\dfrac{1}{0.5}\right) \;=\; \mathrm{diag}(0.5,\;2)$',
's18a2_2': [r'$d_M^2(\mathbf{x},\boldsymbol{\mu}_0) \;=\; \dfrac{2.5^2}{2} + \dfrac{0.5^2}{0.5} \;=\; 3.125+0.5 \;=\; 3.625$',
            r'$d_M^2(\mathbf{x},\boldsymbol{\mu}_1) \;=\; \dfrac{(-1.5)^2}{2} + \dfrac{(-1.5)^2}{0.5} \;=\; 1.125+4.5 \;=\; 5.625$'],
's18a2_3': [r'$\mathbf{w} \;=\; \Sigma^{-1}(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0) \;=\; (0.5\times4,\;2\times2) \;=\; (2,\;4)$',
            r'$b \;=\; -\dfrac{1}{2}(\boldsymbol{\mu}_0+\boldsymbol{\mu}_1)^{T}\mathbf{w} \;=\; -\dfrac{1}{2}\,(4\times2+2\times4) \;=\; -8$'],
's18a2_4': [r'$\mathbf{w}^{T}\mathbf{x}+b \;=\; 2\times2.5+4\times0.5-8 \;=\; -1$',
            r'$P(1\,|\,\mathbf{x}) \;=\; \dfrac{1}{1+e^{1}} \;=\; 0.2689$'],
# ---- 18A.3
's18a3_1': r'$N_{\mathrm{LDA}} \;=\; Kp + \dfrac{p(p+1)}{2} + (K-1) \;=\; 80+210+3 \;=\; 293$',
's18a3_2': r'$N_{\mathrm{QDA}} \;=\; Kp + K\dfrac{p(p+1)}{2} + (K-1) \;=\; 80+4\times210+3 \;=\; 923$',
's18a3_3': [r'$N_{\mathrm{LDA}} \;=\; 4\times80+\dfrac{80\times81}{2}+3 \;=\; 320+3240+3 \;=\; 3563$',
            r'$N_{\mathrm{QDA}} \;=\; 320+4\times3240+3 \;=\; 13283$'],
# ---- 18A.4
's18a4_1': r'$J(\mathbf{a}) \;=\; \dfrac{\left(\mathbf{a}^{T}\mathbf{d}\right)^2}{\mathbf{a}^{T}\Sigma\,\mathbf{a}}\,, \qquad \mathbf{d}=(3,\,1)$',
's18a4_2': [r'$\mathbf{a}=(1,0):\quad J=\dfrac{3^2}{0.75}=12$',
            r'$\mathbf{a}=(0,1):\quad J=\dfrac{1^2}{0.75}=1.3333$'],
's18a4_3': [r'$\mathbf{a}^{T}\Sigma\,\mathbf{a} = 0.75\cdot3^2 + 2\cdot0.5\cdot3\cdot1 + 0.75\cdot1^2 = 10.5$',
            r'$J = \dfrac{(3\cdot3+1\cdot1)^2}{10.5} = \dfrac{100}{10.5} = 9.5238$'],
's18a4_4': r'$\mathbf{a}=(7,-3):\quad J=\dfrac{18^2}{22.5}=\dfrac{324}{22.5}=14.4$',
# ---- 18B.2
's18b2_1': r'$\lambda_{\min}(\Sigma_a) \;=\; (1-a)\cdot0 + a\,\dfrac{\mathrm{tr}\,\hat\Sigma}{p} \;=\; a\times0.9331$',
# ---- 18B.3
's18b3_1': r'$b \;=\; -\dfrac{1}{2}(\boldsymbol{\mu}_0+\boldsymbol{\mu}_1)^{T}\mathbf{w} + \ln\dfrac{\pi_1}{\pi_0} \;=\; 52.099 + \ln\dfrac{1-\pi}{\pi}$',
# ---- 18C.2
's18c2_1': [r'$\sum_{k=1}^{3}\pi_k\left(\boldsymbol{\mu}_k-\bar{\boldsymbol{\mu}}\right) \;=\; \mathbf{0}$',
            r'$\boldsymbol{\mu}_3-\bar{\boldsymbol{\mu}} \;=\; -\dfrac{\pi_1(\boldsymbol{\mu}_1-\bar{\boldsymbol{\mu}})+\pi_2(\boldsymbol{\mu}_2-\bar{\boldsymbol{\mu}})}{\pi_3}$'],
's18c2_2': r'$S_b \;=\; \sum_{k=1}^{K}\pi_k\,(\boldsymbol{\mu}_k-\bar{\boldsymbol{\mu}})(\boldsymbol{\mu}_k-\bar{\boldsymbol{\mu}})^{T}\,, \qquad \mathrm{rank}\,S_b \;\leq\; K-1$',
# ---- questions
's18q1_1': r'$\mathrm{error}(t) \;=\; \pi_0\,P(x>t\,|\,0) + \pi_1\,P(x<t\,|\,1)$',
's18q3_1': [r'$\mathrm{LDA}:\quad \mathbf{w}=\Sigma^{-1}(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)$ from class means and one covariance',
            r'$\mathrm{logistic}:\quad \mathbf{w}\ \mathrm{maximises}\ \prod_i P(y_i\,|\,\mathbf{x}_i)$'],
's18q4_1': [r'$w_{\mathrm{LDA}} \;=\; \dfrac{\mu_1-\mu_0}{\sigma^2} \;=\; \dfrac{0.9-0.2}{0.125} \;=\; 5.6$',
            r'$w_{\mathrm{true}} \;=\; \ln\dfrac{0.9/0.1}{0.2/0.8} \;=\; 3.58$'],
}
mkeq(E, '.'); print('solution equations ok:', len(E))
for k in E:
    with Image.open(k + '.png') as im:
        if im.size[0] > 1250: print('  too wide (will be shrunk):', k, im.size[0], 'px')
