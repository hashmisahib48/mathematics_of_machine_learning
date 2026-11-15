# Equation images of the Chapter 49 solutions. Run: python3 code/eqs.py  (writes sol/ch49/s49*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 49A.1
's49a1_1': [r'$\bar{x} \;=\; \dfrac{2+4+6+8+30}{5} \;=\; 10$',
             r'$\sigma^2 \;=\; \dfrac{8^2+6^2+4^2+2^2+20^2}{5} \;=\; \dfrac{520}{5} \;=\; 104\,,\qquad \sigma \;=\; \sqrt{104} \;=\; 10.198$'],
's49a1_2': r'$\tilde{x}_{12} \;=\; \dfrac{12-10}{10.198} \;=\; 0.196\,,\qquad \tilde{x}_{40} \;=\; \dfrac{40-10}{10.198} \;=\; 2.942$',
's49a1_3': r'$\tilde{x}_{12} \;=\; \dfrac{12-2}{30-2} \;=\; \dfrac{10}{28} \;=\; 0.357\,,\qquad \tilde{x}_{40} \;=\; \dfrac{40-2}{28} \;=\; \dfrac{38}{28} \;=\; 1.357$',
's49a1_4': r'$q_{25}=4\,,\quad \mathrm{median}=6\,,\quad q_{75}=8\,,\qquad s \;=\; q_{75}-q_{25} \;=\; 4$',
's49a1_5': r'$\tilde{x}_{12} \;=\; \dfrac{12-6}{4} \;=\; 1.5\,,\qquad \tilde{x}_{40} \;=\; \dfrac{40-6}{4} \;=\; 8.5$',
# ---- 49A.2
's49a2_1': r'$\lambda=0:\qquad \psi(x;0) \;=\; \ln(x+1)\;\;(x\geq 0)\,,\qquad \psi(x;0) \;=\; -\dfrac{(1-x)^{2}-1}{2}\;\;(x<0)$',
's49a2_2': r'$\lambda=2:\qquad \psi(x;2) \;=\; \dfrac{(x+1)^{2}-1}{2}\;\;(x\geq 0)\,,\qquad \psi(x;2) \;=\; -\ln(1-x)\;\;(x<0)$',
's49a2_3': r'$\psi(3;2) \;=\; \dfrac{4^{2}-1}{2} \;=\; 7.5 \;=\; -\left[-\dfrac{(1-(-3))^{2}-1}{2}\right] \;=\; -\psi(-3;0)$',
# ---- 49C.2
's49c2_1': [r'$\dfrac{a+b}{2} \;=\; \dfrac{2.2781+4.0240}{2} \;=\; 3.151$',
             r'$\sqrt{a\,b} \;=\; \sqrt{2.2781\times 4.0240} \;=\; 3.028$'],
# ---- questions
's49q2_1': r'$\psi(x;\lambda) \;=\; \dfrac{(x+1)^{\lambda}-1}{\lambda} \;\longrightarrow\; \ln(1+x)\,,\qquad \dfrac{d\psi}{dx} \;=\; (x+1)^{\lambda-1}$',
}
mkeq(E, '.'); print('solution equations ok:', len(E))
