# Equation images of Chapter 1. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq
import matplotlib.pyplot as plt
T=r'^{\mathrm{T}}'; X=r'\mathbf{X}'; w=r'\mathbf{w}'
E={
'e131': rf'$\hat{{y}} \;=\; f(\mathbf{{x}};\,{w},\,b)$',
'e132': r'$\hat{y}_i \;=\; b + w_1 x_{i1} + w_2 x_{i2} + \cdots + w_p x_{ip}$',
'e133': rf'$\hat{{\mathbf{{y}}}} \;=\; {X}{w} + b\,\mathbf{{1}}$',
'e141': r'$r_i \;=\; y_i-\hat{y}_i$',
'e142': r'$\ell_{\mathrm{sq}}(r) \;=\; r^2\,, \qquad \ell_{\mathrm{abs}}(r) \;=\; |r|$',
'e143': rf'$J({w},b) \;=\; \dfrac{{1}}{{n}}\sum_{{i=1}}^{{n}}\,\ell(r_i)$',
'e144': r'$J(c) \;=\; \dfrac{1}{n}\sum_{i=1}^{n}\,(y_i-c)^2$',
'e146': r'$J(c) \;=\; s^2 + (c-\bar{y})^2\,, \qquad s^2 \;=\; \dfrac{1}{n}\sum_{i=1}^{n}\,(y_i-\bar{y})^2$',
'e147': r'$\dfrac{\mathrm{d}J_{\mathrm{abs}}}{\mathrm{d}c} \;=\; \dfrac{N_{\mathrm{below}}-N_{\mathrm{above}}}{n}$',
'e151': rf'${w}_{{t+1}} \;=\; {w}_t \;-\; \eta\,\nabla J({w}_t)$',
'e152': r'$\dfrac{\mathrm{d}J}{\mathrm{d}c} \;=\; 2\,(c-\bar{y})$',
'e161': rf'$J_{{\mathrm{{train}}}} \;=\; \dfrac{{1}}{{n}}\sum_{{i\in \mathrm{{train}}}}\,\ell(r_i)\,, \qquad J_{{\mathrm{{test}}}} \;=\; \dfrac{{1}}{{m}}\sum_{{i\in \mathrm{{test}}}}\,\ell(r_i)$',
}
mkeq(E,'.')
def ml(name,lines,ls=2.7):
    fig=plt.figure(figsize=(.01,.01)); fig.text(0,0,'\n'.join(lines),fontsize=11.5,linespacing=ls); fig.savefig(name+'.png',dpi=300,bbox_inches='tight',pad_inches=0.02,facecolor='white'); plt.close(fig)
ml('e145',[r'$\sum_i (y_i-c)^2 \;=\; \sum_i \,[\,(y_i-\bar{y}) + (\bar{y}-c)\,]^2$',
           r'$\;=\; \sum_i (y_i-\bar{y})^2 \;+\; 2(\bar{y}-c)\sum_i (y_i-\bar{y}) \;+\; n\,(\bar{y}-c)^2$',
           r'$\;=\; n\,s^2 \;+\; 0 \;+\; n\,(c-\bar{y})^2$'],2.4)
ml('e153',[r'$c_{t+1} \;=\; c_t - \eta\cdot 2\,(c_t-\bar{y})$',
           r'$c_{t+1}-\bar{y} \;=\; (1-2\eta)\,(c_t-\bar{y})$'],2.6)
print('eqs ok')
