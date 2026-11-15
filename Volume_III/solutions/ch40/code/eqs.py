# Equation images of the Chapter 40 solutions. Run: python3 code/eqs.py  (writes sol/ch40/s40*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 40A.1
's40a1_1': r'$\gamma_{ik}\;=\;\dfrac{\pi_k\,\mathcal{N}(x_i\mid\mu_k,1)}{\sum_{j}\pi_j\,\mathcal{N}(x_i\mid\mu_j,1)}\,,\qquad \mathcal{N}(x\mid\mu,1)=\dfrac{1}{\sqrt{2\pi}}\,e^{-(x-\mu)^{2}/2}$',
's40a1_2': r'$\mathcal{N}(2\mid0,1)=e^{-2}/\sqrt{2\pi}=0.05399\,,\qquad \mathcal{N}(2\mid3,1)=e^{-1/2}/\sqrt{2\pi}=0.24197$',
's40a1_3': [r'$\pi_1\mathcal{N}_1=0.4\times0.05399=0.021596$',
            r'$\pi_2\mathcal{N}_2=0.6\times0.24197=0.145182$'],
's40a1_4': [r'$\gamma_{1}=\dfrac{0.021596}{0.021596+0.145182}=\dfrac{0.021596}{0.166778}=0.1295$',
            r'$\gamma_{2}=1-\gamma_{1}=0.8705$'],
's40a1_5': r'$\mathcal{N}(1\mid0,1)=e^{-1/2}/\sqrt{2\pi}=0.24197\,,\qquad \mathcal{N}(1\mid3,1)=e^{-2}/\sqrt{2\pi}=0.05399$',
's40a1_6': [r'$\pi_1\mathcal{N}_1=0.4\times0.24197=0.096788$',
            r'$\pi_2\mathcal{N}_2=0.6\times0.05399=0.032395$'],
's40a1_7': [r'$\gamma_{1}=\dfrac{0.096788}{0.096788+0.032395}=\dfrac{0.096788}{0.129183}=0.7492$',
            r'$\gamma_{2}=1-\gamma_{1}=0.2508$'],
# ---- 40A.2
's40a2_1': [r'$N_1=0.9+0.8+0.2+0.1=2.0\,,\qquad N_2=0.1+0.2+0.8+0.9=2.0$',
            r'$\pi_1=\dfrac{N_1}{4}=0.5\,,\qquad \pi_2=\dfrac{N_2}{4}=0.5$'],
's40a2_2': r'$\mu_1=\dfrac{0.9\cdot1+0.8\cdot2+0.2\cdot6+0.1\cdot7}{2.0}=\dfrac{4.4}{2.0}=2.2$',
's40a2_3': r'$\mu_2=\dfrac{0.1\cdot1+0.2\cdot2+0.8\cdot6+0.9\cdot7}{2.0}=\dfrac{11.6}{2.0}=5.8$',
's40a2_4': r'$\sigma_1^{2}=\dfrac{0.9(1-2.2)^{2}+0.8(2-2.2)^{2}+0.2(6-2.2)^{2}+0.1(7-2.2)^{2}}{2.0}$',
's40a2_5': r'$=\;\dfrac{1.296+0.032+2.888+2.304}{2.0}=\dfrac{6.52}{2.0}=3.26$',
's40a2_6': r'$\sigma_2^{2}=\dfrac{0.1(1-5.8)^{2}+0.2(2-5.8)^{2}+0.8(6-5.8)^{2}+0.9(7-5.8)^{2}}{2.0}$',
's40a2_7': r'$=\;\dfrac{2.304+2.888+0.032+1.296}{2.0}=\dfrac{6.52}{2.0}=3.26$',
's40a2_8': r'$N_1\mu_1+N_2\mu_2=2.0\times2.2+2.0\times5.8=16=1+2+6+7$',
# ---- 40A.3
's40a3_1': r'$q=(K-1)+Kp+c\,,\qquad \mathrm{BIC}=-2\hat\ell+q\ln n\,,\qquad \mathrm{AIC}=-2\hat\ell+2q$',
's40a3_2': [r'$c=\dfrac{Kp(p+1)}{2}=30\ \ (\mathrm{full})\,,\qquad c=Kp=12\ \ (\mathrm{diag})$',
            r'$c=K=3\ \ (\mathrm{spherical})$'],
's40a3_3': r'$\ln 500=6.2146\,,\qquad 2\ln500=12.43$',
# ---- 40C.1
's40c1_1': [r'$\ell\;=\;\ln\left(\dfrac{1}{2\,\sigma_2\sqrt{2\pi}}+\dfrac{1}{2}\,\mathcal{N}(170\mid160,100)\right)$',
            r'$\quad+\;\sum_{i\neq4}\ln\left(\dfrac{1}{2}\mathcal{N}(x_i\mid160,100)+\dfrac{1}{2}\mathcal{N}(x_i\mid170,\sigma_2^{2})\right)$'],
's40c1_2': r'$\ell\;\approx\;-\ln\sigma_2+\mathrm{const}\;\longrightarrow\;\infty\qquad(\sigma_2\to0)$',
# ---- questions
's40q2_1': [r'$P(\mathrm{at\ least\ one\ start\ reaches\ the\ best})\;=\;1-(1-p)^{n_{\mathrm{init}}}$',
            r'$p=\dfrac{23}{30}=0.767$'],
's40q3_1': r'$\mathrm{BIC:}\ \ \Delta\hat\ell>\dfrac{\Delta q\,\ln n}{2}\,,\qquad \mathrm{AIC:}\ \ \Delta\hat\ell>\Delta q$',
}
mkeq(E, '.')
print('eqs ok')
