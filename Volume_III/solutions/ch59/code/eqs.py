# Equation images of the Chapter 59 solutions. Run: python3 code/eqs.py  (writes sol/ch59/s59*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 59A.1
's59a1_1': r'$\mathrm{fits}_{\mathrm{search}} \;=\; N_{\mathrm{cand}}\,K_{\mathrm{in}}\,n_{\mathrm{rep}} + 1 \;=\; 40\times5\times2+1 \;=\; 401$',
's59a1_2': r'$\mathrm{fits}_{\mathrm{nested}} \;=\; K_{\mathrm{out}}\left(N_{\mathrm{cand}}\,K_{\mathrm{in}}\,n_{\mathrm{rep}}+1\right) \;=\; 5\times401 \;=\; 2005$',
# ---- 59A.2
's59a2_1': r'$\hat{a} \;=\; \dfrac{143-12}{143} \;=\; \dfrac{131}{143} \;=\; 0.9161$',
's59a2_2': r'$\sqrt{\dfrac{\hat{a}\,(1-\hat{a})}{n}} \;=\; \sqrt{\dfrac{0.9161\times0.0839}{143}} \;=\; \sqrt{0.000538} \;=\; 0.0232$',
's59a2_3': r'$0.9161 \;\pm\; 1.96\times0.0232 \;=\; 0.9161 \;\pm\; 0.0454 \;=\; [\,0.871\,,\;0.962\,]$',
}
mkeq(E, '.'); print('solution equations ok:', len(E))
