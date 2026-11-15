# Equation images of the Chapter 41 solutions. Run: python3 code/eqs.py  (writes sol/ch41/s41*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 41A.1
's41a1_1': r'$a(p) \;=\; \dfrac{|3-1|+|3-2|}{2} \;=\; \dfrac{2+1}{2} \;=\; 1.5$',
's41a1_2': r'$\bar d(p,C_2) \;=\; \dfrac{|3-5|+|3-6|+|3-8|}{3} \;=\; \dfrac{2+3+5}{3} \;=\; 3.333$',
's41a1_3': r'$\bar d(p,C_3) \;=\; \dfrac{|3-10|+|3-11|}{2} \;=\; \dfrac{7+8}{2} \;=\; 7.5$',
's41a1_4': r'$s(p) \;=\; \dfrac{b-a}{\max(a,b)} \;=\; \dfrac{3.333-1.5}{3.333} \;=\; \dfrac{1.833}{3.333} \;=\; 0.55$',
# ---- 41A.2
's41a2_1': r'$P_{11} \;=\; \binom{4}{2}+\binom{1}{2}+\binom{1}{2}+\binom{4}{2} \;=\; 6+0+0+6 \;=\; 12$',
's41a2_2': [r'$\sum_i\binom{n_{i\cdot}}{2} \;=\; \binom{5}{2}+\binom{5}{2} \;=\; 20\,,\qquad \sum_j\binom{n_{\cdot j}}{2} \;=\; 10+10 \;=\; 20$', r'$\binom{n}{2} \;=\; \binom{10}{2} \;=\; 45$'],
's41a2_3': [r'$P_{10}=20-12=8\,,\qquad P_{01}=20-12=8$', r'$P_{00}=45-12-8-8=17$'],
's41a2_4': r'$\mathrm{RI} \;=\; \dfrac{P_{11}+P_{00}}{\binom{n}{2}} \;=\; \dfrac{12+17}{45} \;=\; 0.6444$',
's41a2_5': r'$\mathrm{E} \;=\; \dfrac{20\times 20}{45} \;=\; 8.889\,,\qquad \dfrac{20+20}{2} \;=\; 20$',
's41a2_6': r'$\mathrm{ARI} \;=\; \dfrac{12-8.889}{20-8.889} \;=\; \dfrac{3.111}{11.111} \;=\; 0.28$',
's41a2_7': r'$\mathrm{ARI} \;=\; \dfrac{2\,(12\times17-8\times8)}{20\times25+20\times25} \;=\; \dfrac{280}{1000} \;=\; 0.28$',
# ---- 41A.3
's41a3_1': r'$\bar x=\dfrac{1+2+3+7+8+9}{6}=5\,,\qquad \mu_1=2\,,\quad \mu_2=8$',
's41a3_2': r'$\mathrm{SSB} \;=\; 3\,(2-5)^2+3\,(8-5)^2 \;=\; 27+27 \;=\; 54$',
's41a3_3': [r'$\mathrm{SSW} \;=\; (1-2)^2+0+(3-2)^2+(7-8)^2+0+(9-8)^2$', r'$\qquad\quad =\; 2+2 \;=\; 4$'],
's41a3_4': r'$\mathrm{CH} \;=\; \dfrac{\mathrm{SSB}/(K-1)}{\mathrm{SSW}/(n-K)} \;=\; \dfrac{54/1}{4/4} \;=\; 54$',
's41a3_5': r'$S_1 \;=\; \dfrac{|1-2|+|2-2|+|3-2|}{3} \;=\; \dfrac{2}{3}\,,\qquad S_2 \;=\; \dfrac{2}{3}$',
's41a3_6': r'$\dfrac{S_1+S_2}{\Vert\mu_1-\mu_2\Vert} \;=\; \dfrac{4/3}{6} \;=\; 0.2222$',
's41b2_1': r'$\mathrm{E[RI]} \;=\; 0.329\times\dfrac{1}{30}+0.671\times\dfrac{29}{30} \;=\; 0.0110+0.6487 \;=\; 0.660$',
# ---- questions
's41q3_1': r'$\mathrm{RI}_{\max} \;=\; 1-\dfrac{\sum_i\binom{n_{i\cdot}}{2}}{\binom{n}{2}}$',
's41q3_2': r'$1-\dfrac{3\times1225}{11175} \;=\; 0.6711\,,\qquad 1-\dfrac{2\times2775}{11175} \;=\; 0.5034$',
's41q2_1': r'$\mathrm{ARI} \;=\; \dfrac{2\,(P_{11}P_{00}-P_{10}P_{01})}{(P_{11}+P_{10})(P_{10}+P_{00})+(P_{11}+P_{01})(P_{01}+P_{00})}$',
}
mkeq(E, '.'); print('solution equations ok:', len(E))
