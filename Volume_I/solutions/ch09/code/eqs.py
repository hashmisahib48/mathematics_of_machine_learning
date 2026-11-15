# Equation images of the Chapter 9 solutions. Run: python3 code/eqs.py  (writes sol/ch09/s9*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- questions
's9q1_1': r'$d_{i}^{\perp} \;=\; \dfrac{|\,y_i-b-w\,x_i\,|}{\sqrt{1+w^{2}}}$',
's9q2_1': r'$J_{1}(b,w) \;=\; \sum_{i=1}^{n}\,|\,y_i-b-w\,x_i\,|$',
# ---- 9A.1
's9a1_1': r'$\bar{x} \;=\; \dfrac{0+1+2+3}{4} \;=\; 1.5\,,\qquad \bar{y} \;=\; \dfrac{1+3+4+6}{4} \;=\; 3.5$',
's9a1_2': [r'$S_{xy} \;=\; 3.75+0.25+0.25+3.75 \;=\; 8$', r'$S_{xx} \;=\; 2.25+0.25+0.25+2.25 \;=\; 5$'],
's9a1_3': r'$w \;=\; \dfrac{S_{xy}}{S_{xx}} \;=\; \dfrac{8}{5} \;=\; 1.6\,,\qquad b \;=\; \bar{y}-w\,\bar{x} \;=\; 3.5-1.6\times 1.5 \;=\; 1.1$',
's9a1_4': r'$\mathrm{SSE} \;=\; (-0.1)^2+0.3^{2}+(-0.3)^2+0.1^{2} \;=\; 0.01+0.09+0.09+0.01 \;=\; 0.20$',
# ---- 9A.2
's9a2_1': r'$\mathrm{SSE} \;=\; 0.16+0.04+0.04+0.16+1.00 \;=\; 1.40$',
's9a2_2': [r'$\mathrm{SSE}-\mathrm{SSE}_{\min} \;=\; \sum_{i}\left[(1-0.9)+(1.6-1.7)\,x_i\right]^{2}$', r'$=\; 0.01\,(0+1+4+9+16) \;=\; 0.30$'],
# ---- 9A.3
's9a3_1': r'$J(w) \;=\; \sum_{i=1}^{n}\,(y_i-w\,x_i)^{2}$',
's9a3_2': r'$\dfrac{dJ}{dw} \;=\; -2\sum_i x_i\,(y_i-w\,x_i) \;=\; 0 \quad\Longrightarrow\quad w \;=\; \dfrac{\sum_i x_i y_i}{\sum_i x_i^{2}}$',
's9a3_3': [r'$\sum x_i y_i \;=\; 3+8+18+28+50 \;=\; 107\,,\qquad \sum x_i^{2} \;=\; 1+4+9+16+25 \;=\; 55$', r'$w \;=\; \dfrac{107}{55} \;=\; 1.9455$'],
's9a3_4': r'$\mathrm{SSE} \;=\; \sum y_i^{2}-\dfrac{(\sum x_i y_i)^{2}}{\sum x_i^{2}} \;=\; 210-\dfrac{107^{2}}{55} \;=\; 210-208.164 \;=\; 1.836$',
's9a3_5': r'$\mathrm{SSE}_{0}-\mathrm{SSE} \;=\; \sum_i\left[\,0.9+(1.7-1.9455)\,x_i\,\right]^{2} \;=\; 0.736 \;=\; 1.836-1.100$',
# ---- 9A.4
's9a4_1': r'$\mathrm{SST} \;=\; (-2.5)^2+(-0.5)^2+0.5^{2}+2.5^{2} \;=\; 6.25+0.25+0.25+6.25 \;=\; 13$',
's9a4_2': r'$R^{2} \;=\; 1-\dfrac{\mathrm{SSE}}{\mathrm{SST}} \;=\; 1-\dfrac{0.20}{13} \;=\; 0.9846$',
's9a4_3': [r'$\mathrm{SST}-\mathrm{SSE} \;=\; 12.8 \;=\; w^{2}\,S_{xx} \;=\; 1.6^{2}\times 5$', r'$R^{2} \;=\; \dfrac{S_{xy}^{2}}{S_{xx}\,\mathrm{SST}} \;=\; \dfrac{64}{5\times 13} \;=\; \dfrac{64}{65}$'],
# ---- 9B.2
's9b2_1': r'$S_{xy} \;=\; \sum_i (x_i-\bar{x})\,y_i \;=\; (-2)(3)+(-1)(4)+0\cdot 6+(1)(7)+2\,y_5 \;=\; 2\,y_5-3$',
's9b2_2': r'$w \;=\; \dfrac{2\,y_5-3}{10} \;=\; -0.3+0.2\,y_5 \;=\; 1.7+0.2\,(y_5-10)$',
's9b2_3': r'$b \;=\; \bar{y}-3\,w \;=\; \dfrac{20+y_5}{5}-3\,(-0.3+0.2\,y_5) \;=\; 4.9-0.4\,y_5$',
# ---- 9C.1
's9c1_1': r'$\sum_i (y_i-b-w\,x_i) \;=\; 0 \qquad\mathrm{and}\qquad \sum_i x_i\,(y_i-b-w\,x_i) \;=\; 0$',
's9c1_2': r'$\sum_i r_i \;=\; \sum_i (y_i-w\,x_i) \;=\; n\,(\bar{y}-w\,\bar{x}) \;=\; 5\,(6-1.9455\times 3) \;=\; 0.818$',
# ---- 9C.2
's9c2_1': r'$R^{2}_{\mathrm{train}} \;\approx\; 1-\dfrac{\sigma^{2}\,(n-p-1)}{(n-1)\,\mathrm{Var}(y)} \;=\; 1-\dfrac{4\times 1}{59\times 9} \;=\; 0.992 \qquad (p=58)$',
's9c2_2': r'$R^{2}_{\mathrm{adj}} \;=\; 1-(1-R^{2})\,\dfrac{n-1}{n-p-1} \;=\; 1-0.03\times\dfrac{59}{1} \;=\; -0.77$',
's9c2_3': r'$E\left[\mathrm{MSE}_{\mathrm{test}}\right] \;=\; \sigma^{2}\left(1+\dfrac{p}{n-p-2}\right)$',
# ---- 9C.3
's9c3_1': [r'$\hat{y} \;=\; b+w_{\mathrm{km}}\,x_{\mathrm{km}} \;=\; b+\dfrac{w_{\mathrm{km}}}{1000}\,(1000\,x_{\mathrm{km}}) \;=\; b+w_{\mathrm{m}}\,x_{\mathrm{m}}$', r'$w_{\mathrm{m}} \;=\; \dfrac{1.7}{1000} \;=\; 0.0017$'],
's9c3_2': r'$\tilde{w}_{j} \;=\; w_{j}\,s_{j}\,,\qquad s_{j}=\sqrt{\dfrac{1}{n}\sum_i (x_{ij}-\bar{x}_{j})^{2}}$',
}
mkeq(E, '.')
from PIL import Image
for k in sorted(E):
    w = Image.open(f'{k}.png').size[0] / 300
    if w > 5.2: print('TOO WIDE', k, round(w, 2))
print(len(E), 'equation images written')
