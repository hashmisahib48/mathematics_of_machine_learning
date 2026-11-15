# Equation images of the Chapter 26 solutions. Run: python3 code/eqs.py  (writes sol/ch26/s26*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 26A.1
's26a1_1': r'$\varepsilon \;=\; w_2+w_5 \;=\; \dfrac{1}{5}+\dfrac{1}{5} \;=\; 0.4$',
's26a1_2': r'$\alpha \;=\; \ln\dfrac{1-\varepsilon}{\varepsilon} \;=\; \ln\dfrac{0.6}{0.4} \;=\; \ln 1.5 \;=\; 0.4055\,,\qquad e^{\alpha}=1.5$',
's26a1_3': r'$Z \;=\; 3\times 0.2+2\times 0.2\times 1.5 \;=\; 0.6+0.6 \;=\; 1.2$',
's26a1_4': r'$w^{\mathrm{new}}_{\mathrm{mistake}} \;=\; \dfrac{0.3}{1.2} \;=\; 0.25\,,\qquad w^{\mathrm{new}}_{\mathrm{other}} \;=\; \dfrac{0.2}{1.2} \;=\; 0.1667$',
's26a1_5': r'$\varepsilon^{\mathrm{new}} \;=\; 0.25+0.25 \;=\; 0.5$',
# ---- 26A.2
's26a2_1': r'$\alpha \;=\; \eta\,\left[\ln\dfrac{1-\varepsilon}{\varepsilon}+\ln(K-1)\right] \;=\; \eta\,\left[\ln 1+\ln 2\right] \;=\; 0.6931\,\eta$',
's26a2_2': r'$\eta=1:\quad e^{\alpha}=2\,,\qquad Z \;=\; 3\times\dfrac{1}{6}+3\times\dfrac{2}{6} \;=\; 1.5$',
's26a2_3': [r'$w_{\mathrm{mistake}} \;=\; \dfrac{2/6}{1.5} \;=\; 0.2222\,,\qquad w_{\mathrm{other}} \;=\; \dfrac{1/6}{1.5} \;=\; 0.1111$', r'$\mathrm{total\ weight\ of\ the\ mistakes:}\ \ 3\times 0.2222 \;=\; 0.6667$'],
's26a2_4': [r'$\eta=0.5:\quad \alpha=0.3466\,,\quad e^{\alpha}=\sqrt{2}=1.4142$', r'$Z \;=\; \dfrac{3\times 1.4142+3}{6} \;=\; 1.2071$'],
's26a2_5': [r'$w_{\mathrm{mistake}} \;=\; \dfrac{1.4142/6}{1.2071} \;=\; 0.1953\,,\qquad w_{\mathrm{other}} \;=\; \dfrac{1/6}{1.2071} \;=\; 0.1381$', r'$\mathrm{total\ weight\ of\ the\ mistakes:}\ \ 3\times 0.1953 \;=\; 0.5858$'],
# ---- 26A.3
's26a3_1': r'$\sum_m \alpha_m h_m \;=\; 1.0\times(+1)+0.6\times(-1)+0.5\times(-1) \;=\; -0.1$',
's26a3_2': r'$f \;=\; \dfrac{2\sum_m \alpha_m h_m}{\sum_m \alpha_m} \;=\; \dfrac{2\times(-0.1)}{1.0+0.6+0.5} \;=\; \dfrac{-0.2}{2.1} \;=\; -0.0952$',
's26a3_3': r'$P(+1) \;=\; \dfrac{1}{1+e^{-f}} \;=\; \dfrac{1}{1+e^{0.0952}} \;=\; 0.4762$',
# ---- 26A.4
's26a4_1': [r'$D \;=\; \max_i |y_i-h(x_i)| \;=\; 3$', r'$L_i=\dfrac{|y_i-h(x_i)|}{D}:\quad 0\,,\ \dfrac{1}{3}\,,\ \dfrac{1}{3}\,,\ 1$'],
's26a4_2': r'$\bar{\varepsilon} \;=\; \dfrac{1}{4}\left(0+\dfrac{1}{3}+\dfrac{1}{3}+1\right) \;=\; \dfrac{5}{12} \;=\; 0.4167$',
's26a4_3': [r'$\beta \;=\; \dfrac{\bar{\varepsilon}}{1-\bar{\varepsilon}} \;=\; \dfrac{5/12}{7/12} \;=\; \dfrac{5}{7} \;=\; 0.7143$', r'$\alpha \;=\; \ln\dfrac{1}{\beta} \;=\; \ln 1.4 \;=\; 0.3365$'],
's26a4_4': r'$Z \;=\; 0.1786+0.1998+0.1998+0.2500 \;=\; 0.8281$',
's26a4_5': r'$\bar{\varepsilon}_{\mathrm{square}} \;=\; \dfrac{1}{4}\left(0+\dfrac{1}{9}+\dfrac{1}{9}+1\right) \;=\; 0.3056$',
's26a4_6': [r'$\bar{\varepsilon}_{\mathrm{exp}} \;=\; \dfrac{1}{4}\left[0+2\,(1-e^{-1/3})+(1-e^{-1})\right]$', r'$\;=\; \dfrac{0.5669+0.6321}{4} \;=\; 0.2998$'],
# ---- questions
's26q1_1': r'$\alpha(1-\varepsilon) \;=\; \ln\dfrac{\varepsilon}{1-\varepsilon} \;=\; -\ln\dfrac{1-\varepsilon}{\varepsilon} \;=\; -\alpha(\varepsilon)$',
's26q1_2': r'$\varepsilon_m^{\mathrm{new}} \;=\; \dfrac{\varepsilon_m\, e^{\alpha_m}}{\varepsilon_m\, e^{\alpha_m}+(1-\varepsilon_m)} \;=\; \dfrac{1-\varepsilon_m}{2\,(1-\varepsilon_m)} \;=\; \dfrac{1}{2}$',
}
mkeq(E, '.'); print('solution equations ok:', len(E))
