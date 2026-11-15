# Equation and matrix images of Chapter 24. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq, matimg
import matplotlib.pyplot as plt
E={
'e2431': r'$f(\mathbf{x}) \;=\; \sum_{t=1}^{T} c_t\;\mathbf{1}\,[\,\mathbf{x}\in R_t\,]$',
'e2432': r'$p_{tk} \;=\; \dfrac{n_{tk}}{n_t}\,, \qquad k = 1,\dots,K$',
'e2433': r'$G(t) \;=\; 1-\sum_{k=1}^{K} p_{tk}^{\,2} \;=\; \sum_{k=1}^{K} p_{tk}\,(1-p_{tk})$',
'e2434': r'$H(t) \;=\; -\sum_{k=1}^{K} p_{tk}\,\log_2 p_{tk}\,, \qquad M(t) \;=\; 1-\max_k\, p_{tk}$',
'e2435': r'$I_{\mathrm{split}} \;=\; \dfrac{n_L}{n_t}\,I(L) \;+\; \dfrac{n_R}{n_t}\,I(R)$',
'e2436': r'$\Delta \;=\; I(t) \;-\; \dfrac{n_L}{n_t}\,I(L) \;-\; \dfrac{n_R}{n_t}\,I(R)$',
'e2437': r'$s \;=\; \dfrac{v_{(i)}+v_{(i+1)}}{2}\,, \qquad (j^{*},s^{*}) \;=\; \arg\max_{j,\,s}\ \Delta(j,s)$',
'e2438': r'$\dfrac{N_t}{N}\left(I(t)\;-\;\dfrac{N_{tR}}{N_t}\,I(R)\;-\;\dfrac{N_{tL}}{N_t}\,I(L)\right)\ \geq\ \mathtt{min\_impurity\_decrease}$',
'e2441': r'$c_t \;=\; \arg\min_{c}\ \sum_{i\in t}\,(y_i-c)^2 \;=\; \bar{y}_t \;=\; \dfrac{1}{n_t}\sum_{i\in t} y_i$',
'e2442': r'$I(t) \;=\; \dfrac{1}{n_t}\sum_{i\in t}\,(y_i-\bar{y}_t)^2$',
'e2443': r'$I_{\mathrm{abs}}(t) \;=\; \dfrac{1}{n_t}\sum_{i\in t}\,\left|\,y_i-\mathrm{med}_t\,\right|\,, \qquad I_{\mathrm{Poisson}}(t) \;=\; \dfrac{1}{n_t}\sum_{i\in t}\,y_i\ln\dfrac{y_i}{\bar{y}_t}$',
'e2451': r'$R(T) \;=\; \sum_{t\ \mathrm{leaf\ of}\ T} \dfrac{N_t}{N}\,I(t)\,, \qquad R_\alpha(T) \;=\; R(T) + \alpha\,|T|$',
'e2452': r'$R(t)+\alpha \;\leq\; R(T_t)+\alpha\,|T_t| \quad\Longleftrightarrow\quad \alpha \;\geq\; \dfrac{R(t)-R(T_t)}{|T_t|-1}$',
'e2453': r'$\alpha_{\mathrm{eff}}(t) \;=\; \dfrac{R(t)-R(T_t)}{|T_t|-1}\,, \qquad R(t) \;=\; \dfrac{N_t}{N}\,I(t)$',
'e2491': r'$\mathrm{imp}(j) \;=\; \dfrac{\sum_{t:\ \mathrm{split\ on}\ j}\,(N_t/N)\,\Delta_t}{\sum_{t\ \mathrm{internal}}\,(N_t/N)\,\Delta_t}$',
'e2461': r'$G(\mathrm{root}) \;=\; 1-\left(\dfrac{5}{10}\right)^{2}-\left(\dfrac{5}{10}\right)^{2} \;=\; 0.5$',
'e2462': r'$G_{\mathrm{split}} \;=\; \dfrac{4}{10}\cdot 0 \;+\; \dfrac{6}{10}\left[\,1-\left(\dfrac{1}{6}\right)^{2}-\left(\dfrac{5}{6}\right)^{2}\right] \;=\; 0.6\times 0.2778 \;=\; 0.1667$',
'e2463': r'$\Delta \;=\; 0.5-0.1667 \;=\; 0.3333$',
}
mkeq(E,'.')
print('eqs ok')
