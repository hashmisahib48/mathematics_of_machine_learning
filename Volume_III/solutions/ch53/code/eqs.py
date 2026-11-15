# Equation images of the Chapter 53 solutions. Run: python3 code/eqs.py  (writes sol/ch53/s53*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 53A.1
's53a1_1': r'$\bar{x}_0=\dfrac{1+2+3}{3}=2\,,\;\; \bar{x}_1=\dfrac{4+5+6}{3}=5\,,\;\; \bar{x}_2=\dfrac{7+8+9}{3}=8$',
's53a1_7': r'$\bar{x} \;=\; \dfrac{1+2+\cdots+9}{9} \;=\; \dfrac{2+5+8}{3} \;=\; 5$',
's53a1_7': r'$\bar{x} \;=\; \dfrac{1+2+\cdots+9}{9} \;=\; \dfrac{2+5+8}{3} \;=\; 5$',
's53a1_2': r'$\mathrm{SSB} \;=\; 3(2-5)^2+3(5-5)^2+3(8-5)^2 \;=\; 27+0+27 \;=\; 54$',
's53a1_3': r'$\mathrm{SSW} \;=\; 3\times\left[(-1)^2+0^2+1^2\right] \;=\; 3\times 2 \;=\; 6$',
's53a1_4': r'$K-1 \;=\; 3-1 \;=\; 2\,,\qquad n-K \;=\; 9-3 \;=\; 6$',
's53a1_5': r'$F \;=\; \dfrac{54\,/\,2}{6\,/\,6} \;=\; \dfrac{27}{1} \;=\; 27.0$',
's53a1_6': r'$P(F>f) \;=\; \left(1+\dfrac{f}{3}\right)^{-3} \;\Longrightarrow\; p \;=\; (1+9)^{-3} \;=\; 0.001$',
# ---- 53A.2
's53a2_1': r'$O_0=2+2+2+2=8\,,\;\; O_1=1+1+0+0=2\,,\;\; O_2=0+1+0+1=2$',
's53a2_5': r'$\sum_i x_i \;=\; 8+2+2 \;=\; 12$',
's53a2_5': r'$\sum_i x_i \;=\; 8+2+2 \;=\; 12$',
's53a2_2': r'$E_k \;=\; \dfrac{n_k}{n}\sum_{i}x_{i} \;=\; \dfrac{4}{12}\times 12 \;=\; 4\,,\qquad k=0,1,2$',
's53a2_3': r'$\chi^2 \;=\; \dfrac{(8-4)^2}{4}+\dfrac{(2-4)^2}{4}+\dfrac{(2-4)^2}{4} \;=\; 4+1+1 \;=\; 6.0$',
's53a2_4': r'$P(\chi^2_2>c) \;=\; e^{-c/2} \;\Longrightarrow\; p \;=\; e^{-3} \;=\; 0.0498$',
# ---- 53A.3
's53a3_1': r'$\dfrac{\alpha k}{m} \;=\; \dfrac{0.10\,k}{8} \;=\; 0.0125\,k\,,\qquad k=1,\ldots,8$',
's53a3_2': r'$k^{*} \;=\; \max\{k:\ p_{(k)}\leq 0.0125\,k\} \;=\; 5\,,\qquad p_{(k^{*})} \;=\; 0.043$',
's53a3_3': r'$\mathrm{Fwe}:\ p<\dfrac{\alpha}{m}=\dfrac{0.10}{8}=0.0125\,,\qquad \mathrm{Fpr}:\ p<\alpha=0.10$',
# ---- questions
's53q2_1': r'$\mathrm{E}\!\left[\dfrac{V}{\max(R,1)}\right] \;\leq\; \alpha\,\dfrac{m_0}{m}$',
}
mkeq(E, '.')
