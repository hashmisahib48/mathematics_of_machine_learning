# Equation images of the Chapter 37 solutions. Run: python3 code/eqs.py  (writes sol/ch37/s37*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 37A.1  average linkage
's37a1_1': [r'$d(\mathrm{A},\mathrm{BC})=\frac{1}{2}(2.236+3.162)=2.699$',
            r'$d(\mathrm{D},\mathrm{BC})=\frac{1}{2}(6.325+7.280)=6.802$',
            r'$d(\mathrm{E},\mathrm{BC})=\frac{1}{2}(10.000+11.000)=10.500$'],
's37a1_2': [r'$d(\mathrm{D},\mathrm{ABC})=\frac{1\cdot4.123+2\cdot6.802}{3}=5.909$',
            r'$d(\mathrm{E},\mathrm{ABC})=\frac{1\cdot8.062+2\cdot10.500}{3}=9.687$'],
's37a1_3': r'$d(\mathrm{ABC},\mathrm{DE})=\frac{1\cdot5.909+1\cdot9.687}{2}=7.798$',
's37a1_4': r'$\frac{4.123+8.062+6.325+10.000+7.280+11.000}{6}=\frac{46.790}{6}=7.798$',
# ---- 37A.2  Ward
's37a2_1': r'$\Delta(\mathrm{PQ},\mathrm{R})=\frac{2\cdot1}{2+1}\,\Vert(1,0)-(0,4)\Vert^{2}=\frac{2}{3}\cdot17=11.3333$',
's37a2_2': r'$\Delta(\mathrm{PQ},\mathrm{S})=\frac{2}{3}\,\Vert(1,0)-(6,4)\Vert^{2}=\frac{2}{3}\cdot41=27.3333$',
's37a2_3': r'$h_{1}=\sqrt{2\cdot2}=2.000\,,\qquad h_{2}=\sqrt{2\cdot11.3333}=4.761$',
's37a2_4': [r'$\boldsymbol{\mu}_{\mathrm{PQR}}=\left(\frac{2}{3},\frac{4}{3}\right)\,,\qquad \Vert\mathbf{S}-\boldsymbol{\mu}_{\mathrm{PQR}}\Vert^{2}=\left(\frac{16}{3}\right)^{2}+\left(\frac{8}{3}\right)^{2}=\frac{320}{9}$',
            r'$\Delta(\mathrm{PQR},\mathrm{S})=\frac{3\cdot1}{3+1}\cdot\frac{320}{9}=\frac{80}{3}=26.6667\,,\qquad h_{3}=\sqrt{2\cdot26.6667}=7.303$'],
's37a2_5': [r'$\mathrm{SS}_{\mathrm{total}}=\Vert\mathbf{P}-\bar{\mathbf{x}}\Vert^{2}+\dots=8+4+8+20=40\,,\qquad \bar{\mathbf{x}}=(2,2)$',
            r'$2+11.3333+26.6667=40$'],
's37a2_6': r'$d^{2}(\mathrm{PQ},\mathrm{R})=\frac{2}{3}\cdot16+\frac{2}{3}\cdot20-\frac{1}{3}\cdot4=\frac{68}{3}=2\cdot11.3333$',
# ---- 37A.3  clustering features
's37a3_1': [r'$\mathrm{CF}_{1}=\left(2,\ (1+1,\ 1+3),\ 1^{2}+1^{2}+1^{2}+3^{2}\right)=\left(2,\ (2,4),\ 12\right)$',
            r'$\mathrm{CF}_{2}=\left(2,\ (3+3,\ 1+3),\ 3^{2}+1^{2}+3^{2}+3^{2}\right)=\left(2,\ (6,4),\ 28\right)$'],
's37a3_2': r'$\mathrm{CF}_{1}+\mathrm{CF}_{2}=\left(2+2,\ (2+6,\ 4+4),\ 12+28\right)=\left(4,\ (8,8),\ 40\right)$',
's37a3_3': [r'$R^{2}=\frac{SS}{N}-\left\Vert\frac{\mathbf{LS}}{N}\right\Vert^{2}$',
            r'$R_{1}^{2}=\frac{12}{2}-\left\Vert(1,2)\right\Vert^{2}=6-5=1\,,\qquad R_{2}^{2}=\frac{28}{2}-\left\Vert(3,2)\right\Vert^{2}=14-13=1$',
            r'$R_{12}^{2}=\frac{40}{4}-\left\Vert(2,2)\right\Vert^{2}=10-8=2\,,\qquad R_{12}=\sqrt{2}=1.4142$'],
# ---- 37B.2
's37b2_1': r'$d^{2}(\mathrm{13},\mathrm{2})=\frac{1}{2}\cdot4+\frac{1}{2}\cdot3.89-\frac{1}{4}\cdot3.89=2.9725\,,\qquad \sqrt{2.9725}=1.7241$',
# ---- 37C.2
's37c2_1': r'$\Delta=\frac{1\cdot86}{87}\cdot6.427=6.35\qquad\mathrm{(singleton\ and\ the\ rest)}$',
's37c2_2': r'$\Delta=\frac{42\cdot45}{87}\cdot23.849=21.72\cdot23.849=518.1\qquad\mathrm{(Ward\ clusters)}$',
# ---- questions
's37q1_1': r'$\Delta=\frac{n_{A}\,n_{B}}{n_{A}+n_{B}}\,\Vert\boldsymbol{\mu}_{A}-\boldsymbol{\mu}_{B}\Vert^{2}$',
's37q4_1': r'$\Delta(\mathbf{x},C)=\frac{n_{C}}{n_{C}+1}\,\Vert\mathbf{x}-\boldsymbol{\mu}_{C}\Vert^{2}$',
}
mkeq(E, '.'); print('solution equations ok:', len(E))
