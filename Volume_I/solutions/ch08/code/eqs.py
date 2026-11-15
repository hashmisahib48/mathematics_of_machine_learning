# Equation images of the Chapter 8 solutions. Run: python3 code/eqs.py  (writes sol/ch08/s8*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 8A.1
's8a1_1': r'$\mathrm{MSE} \;=\; \dfrac{6}{4} \;=\; 1.5\,,\qquad \mathrm{RMSE} \;=\; \sqrt{1.5} \;=\; 1.2247$',
's8a1_2': r'$\mathrm{MAE} \;=\; \dfrac{1+1+0+2}{4} \;=\; \dfrac{4}{4} \;=\; 1.0$',
's8a1_3': r'$R^2 \;=\; 1-\dfrac{\mathrm{SSE}}{\mathrm{SST}} \;=\; 1-\dfrac{6}{34} \;=\; 0.8235$',
's8a1_4': r'$R^2 \;=\; 1-\dfrac{\mathrm{MSE}}{\mathrm{SST}/n} \;=\; 1-\dfrac{1.5}{34/4} \;=\; 1-\dfrac{1.5}{8.5} \;=\; 0.8235$',
# ---- 8A.2
's8a2_1': r'$P \;=\; \dfrac{40}{40+20} \;=\; 0.667\,,\qquad R \;=\; \dfrac{40}{40+10} \;=\; 0.800$',
's8a2_2': r'$F_1 \;=\; \dfrac{2\times 0.6667\times 0.8}{0.6667+0.8} \;=\; 0.7273 \;=\; \dfrac{2\times 40}{2\times 40+20+10}$',
's8a2_3': r'$\mathrm{accuracy} \;=\; \dfrac{40+930}{1000} \;=\; 0.970$',
's8a2_4': r'$\mathrm{accuracy}_{\mathrm{nobody}} \;=\; \dfrac{0+950}{1000} \;=\; 0.950\,,\qquad \mathrm{recall} \;=\; \dfrac{0}{50} \;=\; 0$',
# ---- 8A.3
's8a3_1': r'$\bar y_{-i} \;=\; \dfrac{n\bar y - y_i}{n-1}\,,\qquad y_i-\bar y_{-i} \;=\; \dfrac{n}{n-1}\,(y_i-\bar y)$',
's8a3_2': r'$\mathrm{MSE}_{\mathrm{LOO}} \;=\; \dfrac{20.25+2.25+36}{3} \;=\; \dfrac{58.5}{3} \;=\; 19.5$',
's8a3_3': r'$\mathrm{MSE}_{\mathrm{train}} \;=\; \dfrac{9+1+16}{3} \;=\; \dfrac{26}{3} \;=\; 8.667$',
's8a3_4': r'$\dfrac{\mathrm{MSE}_{\mathrm{LOO}}}{\mathrm{MSE}_{\mathrm{train}}} \;=\; \left(\dfrac{n}{n-1}\right)^{2} \;=\; \left(\dfrac{3}{2}\right)^{2} \;=\; 2.25 \;=\; \dfrac{19.5}{8.667}$',
# ---- 8A.4
's8a4_1': r'$n_{\mathrm{test}} \;=\; \lceil 0.2\times 150\rceil \;=\; 30\,,\qquad n_{\mathrm{train}} \;=\; 150-30 \;=\; 120$',
's8a4_2': r'$\mathrm{fit:}\ \ 120\times\dfrac{4}{5} \;=\; 96\,,\qquad \mathrm{score:}\ \ 120\times\dfrac{1}{5} \;=\; 24$',
's8a4_3': r'$103 \;=\; 5\times 20+3\,,\qquad 3\times 21+2\times 20 \;=\; 103$',
# ---- 8B.2
's8b2_1': r'$P(\mathrm{correct}) \;=\; \sum_{k} P(y=k)\,P(\hat y=k) \;=\; \sum_{k} p_k\,p_k \;=\; \sum_{k} p_k^{2}$',
's8b2_2': r'$0.1002^{2}+0.8998^{2} \;=\; 0.0100+0.8096 \;=\; 0.8196$',
's8b2_3': r'$\mathrm{Var} \;=\; \sum_{k} q_k\,p_k(1-p_k) \;=\; 450\times 0.09018 \;=\; 40.58$',
's8b2_4': r'$\mathrm{sd}(\mathrm{accuracy}) \;=\; \dfrac{\sqrt{40.58}}{450} \;=\; \dfrac{6.37}{450} \;=\; 0.0142$',
# ---- 8C.1
's8c1_1': r'$R^2 \;=\; 1-\dfrac{\mathrm{SSE}}{\mathrm{SST}} \;=\; 1-\dfrac{11.14}{5.00} \;=\; -1.228$',
's8c1_2': r'$\mathrm{SSE}_{\mathrm{const}} \;=\; 174\,,\qquad 1-\dfrac{11.14}{174} \;=\; 0.936$',
# ---- 8C.3
's8c3_1': r'$F_1 \;=\; \dfrac{2\times 267}{2\times 267+159+0} \;=\; \dfrac{534}{693} \;=\; 0.7706$',
's8c3_2': r'$F_1 \;=\; \dfrac{2\times 0}{2\times 0+0+159} \;=\; 0$',
# ---- questions
's8q4_1': r'$\mathrm{SSE} \;=\; \sum_i (y_i-c)^2 \;=\; \mathrm{SST}+n\,(\bar y-c)^2$',
's8q4_2': r'$R^2 \;=\; 1-\dfrac{\mathrm{SSE}}{\mathrm{SST}} \;=\; -\,\dfrac{n\,(\bar y-c)^2}{\mathrm{SST}}\;\leq\; 0$',
}
mkeq(E, '.'); print('solution equations ok:', len(E))
