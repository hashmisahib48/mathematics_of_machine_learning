# Equation images of the Chapter 24 solutions. Run: python3 code/eqs.py  (writes sol/ch24/s24*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 24A.1
's24a1_1': r'$p_1=\dfrac{6}{12}=0.5000\,,\qquad p_2=\dfrac{4}{12}=0.3333\,,\qquad p_3=\dfrac{2}{12}=0.1667$',
's24a1_2': [r'$G \;=\; 1-\left(0.5000^2+0.3333^2+0.1667^2\right)$',
            r'$\;=\; 1-(0.2500+0.1111+0.0278) \;=\; 0.6111$'],
's24a1_3': [r'$H \;=\; -\left(0.5000\log_2 0.5000+0.3333\log_2 0.3333+0.1667\log_2 0.1667\right)$',
            r'$\;=\; 0.5000+0.5283+0.4308 \;=\; 1.4591\ \mathrm{bits}$'],
's24a1_4': r'$M \;=\; 1-\max_k p_k \;=\; 1-0.5000 \;=\; 0.5000$',
# ---- 24A.2
's24a2_1': r'$I(\mathrm{root}) \;=\; 1-\left(\dfrac{4}{8}\right)^2-\left(\dfrac{4}{8}\right)^2 \;=\; 0.5000$',
's24a2_2': [r'$s=1.5:\quad G_L=0\,,\quad G_R=1-\left(\dfrac{3}{7}\right)^2-\left(\dfrac{4}{7}\right)^2=0.4898$',
            r'$\dfrac{1}{8}\cdot 0+\dfrac{7}{8}\cdot 0.4898 \;=\; 0.4286$'],
's24a2_3': [r'$s=5.5:\quad G_L=1-\left(\dfrac{4}{5}\right)^2-\left(\dfrac{1}{5}\right)^2=0.3200\,,\quad G_R=1-1^2=0$',
            r'$\dfrac{5}{8}\cdot 0.3200+\dfrac{3}{8}\cdot 0 \;=\; 0.2000$'],
's24a2_4': r'$\Delta \;=\; I(\mathrm{root})-I_{\mathrm{split}} \;=\; 0.5000-0.2000 \;=\; 0.3000$',
# ---- 24A.3
's24a3_1': [r'$\bar{y}=\dfrac{2+3+10+11+12}{5}=7.6$',
            r'$\mathrm{SST}=(-5.6)^2+(-4.6)^2+2.4^2+3.4^2+4.4^2$',
            r'$\;=\;31.36+21.16+5.76+11.56+19.36=89.20$'],
's24a3_2': [r'$s=1.5:\quad \mathrm{SSE}_R=(3-9)^2+(10-9)^2+(11-9)^2+(12-9)^2$',
            r'$\;=\;36+1+4+9=50$'],
's24a3_3': [r'$s=2.5:\quad c_L=\dfrac{2+3}{2}=2.5\,,\quad c_R=\dfrac{10+11+12}{3}=11$',
            r'$\mathrm{SSE}=\left[(-0.5)^2+0.5^2\right]+\left[(-1)^2+0^2+1^2\right]=0.5+2=2.5$'],
's24a3_4': r'$R^2 \;=\; 1-\dfrac{\mathrm{SSE}}{\mathrm{SST}} \;=\; 1-\dfrac{2.5}{89.2} \;=\; 0.9720$',
's24a3_5': r'$I(\mathrm{root})=\dfrac{89.2}{5}=17.84\;\longrightarrow\;\dfrac{2.5}{5}=0.50\,,\qquad \Delta=17.34$',
# ---- 24A.4
's24a4_1': [r'$\alpha_{\mathrm{eff}}(t_1) \;=\; \dfrac{R(t_1)-R(T_{t_1})}{|T_{t_1}|-1} \;=\; \dfrac{0.08-(0.02+0.01)}{2-1} \;=\; 0.05$'],
's24a4_2': [r'$\alpha_{\mathrm{eff}}(t_0) \;=\; \dfrac{0.20-(0.04+0.02+0.01)}{3-1} \;=\; \dfrac{0.13}{2} \;=\; 0.065$'],
's24a4_3': r'$\alpha_{\mathrm{eff}}(t_0) \;=\; \dfrac{0.20-(0.08+0.04)}{2-1} \;=\; 0.08$',
's24a4_4': [r'$R_{0.06}(T_0)=0.07+0.06\times 3=0.250$',
            r'$R_{0.06}(T_1)=0.12+0.06\times 2=0.240$',
            r'$R_{0.06}(\mathrm{root})=0.20+0.06\times 1=0.260$'],
# ---- 24C
's24c1_1': r'$\min_i y_i \;\leq\; \hat{f}(x) \;\leq\; \max_i y_i \;=\; 2\cdot 5+1 \;=\; 11 \qquad \mathrm{for\ every}\ x$',
's24c1_2': r'$\mathrm{error}(x) \;=\; (2x+1)-11 \;=\; 2\,(x-5) \qquad\Longrightarrow\qquad \mathrm{error}(8)=6$',
's24c2_1': r'$\mathrm{s.e.} \;=\; \sqrt{\dfrac{0.9133\times 0.0867}{143}} \;=\; 0.0235\,,\qquad \sqrt{2}\times 0.0235 \;=\; 0.0333$',
# ---- questions
's24q1_1': [r'$\Delta_M \;=\; \dfrac{2}{8}-\dfrac{3}{8}\cdot 0-\dfrac{5}{8}\cdot\dfrac{2}{5} \;=\; 0$',
            r'$\Delta_G \;=\; 0.375-\dfrac{3}{8}\cdot 0-\dfrac{5}{8}\cdot 0.48 \;=\; 0.075$'],
's24q4_1': r'$\alpha_{\mathrm{eff}}(\mathrm{root}) \;=\; \dfrac{R(\mathrm{root})-R(T)}{|T|-1} \;=\; \dfrac{0.5-0}{4-1} \;=\; 0.1667$',
}
mkeq(E, '.'); print('solution equations ok:', len(E))
