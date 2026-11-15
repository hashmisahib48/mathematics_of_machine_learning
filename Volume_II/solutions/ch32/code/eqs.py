# Equation images of the Chapter 32 solutions. Run: python3 code/eqs.py  (writes sol/ch32/s32*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 32A.1
's32a1_1': r'$v_{12} \;=\; \dfrac{1\times 6+1\times 2}{1+1} \;=\; 4\,, \qquad W_{12}=2$',
's32a1_2': r'$v \;=\; \dfrac{2\times 4+1\times 3}{2+1} \;=\; \dfrac{11}{3} \;=\; 3.6667\,, \qquad W=3$',
's32a1_3': r'$v \;=\; \dfrac{2\times 4+3\times 3.6667}{2+3} \;=\; \dfrac{8+11}{5} \;=\; 3.8\,, \qquad W=5$',
's32a1_4': [r'$\sum_i w_i f_i \;=\; 5\times 3.8+1\times 8 \;=\; 27$', r'$\sum_i w_i y_i \;=\; 6+2+2\times 4+3+8 \;=\; 27$'],
's32a1_5': [r'$\sum_i w_i\,(y_i-f_i)^2 \;=\; 2.2^2+1.8^2+2\times 0.2^2+0.8^2+0$', r'$=\; 4.84+3.24+0.08+0.64 \;=\; 8.8$'],
# ---- 32A.2
's32a2_1': r'$\bar{o}_1=\dfrac{3}{4}=0.75\,,\qquad \bar{o}_2=\dfrac{2}{6}=0.3333\,,\qquad \bar{o}=\dfrac{3+2}{10}=0.5$',
's32a2_2': r'$\mathrm{REL} \;=\; \dfrac{4\,(0.05)^2+6\,(0.1333)^2}{10} \;=\; \dfrac{0.01+0.1067}{10} \;=\; 0.0117$',
's32a2_3': r'$\mathrm{RES} \;=\; \dfrac{4\,(0.25)^2+6\,(0.1667)^2}{10} \;=\; \dfrac{0.25+0.1667}{10} \;=\; 0.0417$',
's32a2_4': r'$\mathrm{UNC}=0.5\times 0.5=0.25\,,\qquad \mathrm{BS}=0.0117-0.0417+0.25=0.22$',
's32a2_5': [r'$\mathrm{BS} \;=\; \dfrac{3\,(0.8-1)^2+1\,(0.8-0)^2+2\,(0.2-1)^2+4\,(0.2-0)^2}{10}$', r'$=\; \dfrac{0.12+0.64+1.28+0.16}{10} \;=\; 0.22$'],
's32a2_6': r'$\mathrm{BS}_{0.5} \;=\; \dfrac{10\times(0.5)^2}{10} \;=\; 0.25 \;=\; \mathrm{UNC}$',
# ---- 32A.3
's32a3_1': r'$t_{+} \;=\; \dfrac{n_{+}+1}{n_{+}+2} \;=\; \dfrac{3}{4}\,, \qquad t_{-} \;=\; \dfrac{1}{n_{-}+2} \;=\; \dfrac{1}{4}$',
's32a3_2': r'$\left[t(f)-p(f)\right]+\left[t(-f)-p(-f)\right] \;=\; \left[t-p\right]+\left[(1-t)-(1-p)\right] \;=\; 0$',
's32a3_3': [r'$g(0) \;=\; -\sum_i f_i t_i \;=\; -(-0.5-0.75+0.25+1.5) \;=\; -0.5$',
            r'$h(0) \;=\; \frac{1}{4}\sum_i f_i^2 \;=\; \frac{4+1+1+4}{4} \;=\; 2.5\,, \qquad s \;=\; 0-\dfrac{-0.5}{2.5} \;=\; 0.2$'],
's32a3_4': [r'$p_i=\sigma(0.2\,f_i) \;=\; 0.4013,\ 0.4502,\ 0.5498,\ 0.5987$',
            r'$g \;=\; -2(0.4013-0.25)-1(0.4502-0.75)$', r'$\qquad +\,1(0.5498-0.25)+2(0.5987-0.75) \;=\; -0.0056$'],
's32a3_5': r'$A \;=\; -0.2023\,, \qquad P(f=1) \;=\; \dfrac{1}{1+e^{-0.20231}} \;=\; 0.5504$',
# ---- 32C.1, 32C.2
's32c1_1': r'$\mathrm{BS} \;=\; \mathrm{REL}-\mathrm{RES}+\mathrm{UNC} \;=\; 0-0+\bar{o}\,(1-\bar{o}) \;=\; 0.2500$',
's32c2_1': r'$-\ln(2.2\times 10^{-16}) = 36.04\,,\quad -\ln 0.01 = 4.61\,,\quad (1-0)^2 = 1$',
# ---- questions
's32q1_1': [r'$\lambda_1=4-3.667=0.333\,,\quad \lambda_2=0.333+(1-3.667)=-2.333\,,$', r'$\lambda_3=-2.333+(6-3.667)=0$'],
's32q3_1': r'$P=\dfrac{1}{1+e^{Af+B}}=\dfrac{1}{2} \Longleftrightarrow f=-\dfrac{B}{A}\,, \qquad\quad P=\sigma(2\beta f)=\dfrac{1}{2} \Longleftrightarrow f=0$',
}
mkeq(E, '.'); print('solution equations ok:', len(E))
