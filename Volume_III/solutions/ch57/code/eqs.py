# Equation images of the Chapter 57 solutions. Run: python3 code/eqs.py  (writes sol/ch57/s57*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 57A.1
's57a1_1': r'$\mathrm{accuracy}=\dfrac{TP+TN}{n}=\dfrac{30+940}{1000}=0.970$',
's57a1_2': r'$P=\dfrac{TP}{TP+FP}=\dfrac{30}{50}=0.600\,,\qquad R=\dfrac{TP}{TP+FN}=\dfrac{30}{40}=0.750$',
's57a1_3': r'$F_1=\dfrac{2\,TP}{2\,TP+FN+FP}=\dfrac{60}{60+10+20}=\dfrac{60}{90}=0.6667$',
's57a1_4': r'$F_2=\dfrac{5\,TP}{5\,TP+4\,FN+FP}=\dfrac{150}{150+40+20}=\dfrac{150}{210}=0.7143$',
's57a1_5': r'$\mathrm{MCC}=\dfrac{30\cdot 940-20\cdot 10}{\sqrt{50\cdot 40\cdot 960\cdot 950}}=\dfrac{28\,000}{42\,708.3}=0.6556$',
# ---- 57A.2
's57a2_1': r'$\mathrm{AUC}=\dfrac{\mathrm{wins}+\frac{1}{2}\,\mathrm{ties}}{n_{+}\,n_{-}}=\dfrac{6+0}{3\times 3}=\dfrac{6}{9}=0.6667$',
's57a2_2': r'$\mathrm{AP}=\dfrac{1}{3}\,(1)+\dfrac{1}{3}\left(\dfrac{2}{3}\right)+\dfrac{1}{3}\left(\dfrac{3}{5}\right)=\dfrac{1}{3}\,(2.2667)=0.7556$',
# ---- 57A.3
's57a3_1': r'$\mathrm{LR}_{+}=\dfrac{\mathrm{TPR}}{\mathrm{FPR}}=\dfrac{0.90}{1-0.95}=\dfrac{0.90}{0.05}=18$',
's57a3_2': r'$\dfrac{P}{1-P}=18\times\dfrac{0.01}{0.99}=0.1818\,,\qquad P=\dfrac{0.1818}{1.1818}=0.1538$',
's57a3_3': r'$\dfrac{P}{1-P}=18\times\dfrac{0.10}{0.90}=2.000\,,\qquad P=\dfrac{2}{3}=0.6667$',
's57a3_4': r'$\dfrac{P}{1-P}=18\times\dfrac{0.50}{0.50}=18\,,\qquad P=\dfrac{18}{19}=0.9474$',
# ---- 57C.2
's57c2_1': r'$\dfrac{p_{\mathrm{new}}}{1-p_{\mathrm{new}}}=\dfrac{p}{1-p}\cdot\dfrac{0.05/0.95}{0.50/0.50}=0.0526\,\dfrac{p}{1-p}$',
# ---- questions
's57q1_1': r'$\mathrm{accuracy}=\pi\,\mathrm{recall}+(1-\pi)\,\mathrm{specificity}$',
's57q1_2': r'$0.005\times 0.800+0.995\times 0.991=0.990$',
's57q2_1': r'$P=\dfrac{\mathrm{TPR}\,\pi}{\mathrm{TPR}\,\pi+\mathrm{FPR}\,(1-\pi)}=\dfrac{0.6915\,\pi}{0.6915\,\pi+0.0668\,(1-\pi)}$',
's57q3_1': r'$\mathrm{macro}\ F_1=\dfrac{0.8696+0.4706+0.7059}{3}=0.6820$',
}
mkeq(E, '.')
from PIL import Image
for k in E:
    with Image.open(k + '.png') as im: w = im.size[0] / 300 * 96
    if w > 400: print('TOO WIDE', k, round(w))
