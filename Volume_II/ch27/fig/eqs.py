# Equation images of Chapter 27. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq
E={
'e2731': r'$F_M(\mathbf{x}) \;=\; F_0 \;+\; \eta\sum_{m=1}^{M} h_m(\mathbf{x})$',
'e2732': r'$J(F) \;=\; \dfrac{1}{n}\sum_{i=1}^{n}\,\ell(y_i,\,F(\mathbf{x}_i))$',
'e2733': r'$r_{im} \;=\; -\left[\dfrac{\partial\,\ell(y_i,F)}{\partial F}\right]_{F\,=\,F_{m-1}(\mathbf{x}_i)}$',
'e2734': r'$\ell(y_i,\,F_i+\delta_i) \;\approx\; \ell(y_i,\,F_i)\;-\;r_i\,\delta_i$',
'e2735': r'$h_m \;=\; \arg\min_{h}\ \sum_{i=1}^{n}\,(r_{im}-h(\mathbf{x}_i))^2\,, \qquad h_m(\mathbf{x}) \;=\; \sum_{t=1}^{T_m} c_{mt}\,\mathbf{1}\,[\,\mathbf{x}\in R_{mt}\,]$',
'e2736': r'$\gamma_{mt} \;=\; \arg\min_{\gamma}\ \sum_{\mathbf{x}_i\in R_{mt}}\,\ell(y_i,\,F_{m-1}(\mathbf{x}_i)+\gamma)$',
'e2737': r'$F_m(\mathbf{x}) \;=\; F_{m-1}(\mathbf{x}) \;+\; \eta\sum_{t=1}^{T_m}\gamma_{mt}\,\mathbf{1}\,[\,\mathbf{x}\in R_{mt}\,]$',
'e2741': r'$\ell(y,F) \;=\; \frac{1}{2}\,(y-F)^2\,, \qquad r \;=\; -\dfrac{\partial\ell}{\partial F} \;=\; y-F$',
'e2742': r'$\delta \;=\; \mathrm{quantile}_{\alpha}\{\,|y_i-F_{m-1}(\mathbf{x}_i)|\,:\ i=1,\dots,n\,\}$',
'e2743': r'$\gamma_t \;=\; \mathrm{med}_t \;+\; \dfrac{1}{n_t}\sum_{i\in t}\mathrm{sgn}(e_i-\mathrm{med}_t)\,\min(\delta,\ |e_i-\mathrm{med}_t|)$',
'e2751': r'$p \;=\; \sigma(F) \;=\; \dfrac{1}{1+e^{-F}}\,, \qquad F \;=\; \ln\dfrac{p}{1-p}$',
'e2752': r'$\ell(y,F) \;=\; -y\ln p-(1-y)\ln(1-p) \;=\; \ln\left(1+e^{F}\right)-y\,F$',
'e2753': r'$\dfrac{\partial\ell}{\partial F} \;=\; p-y\,, \qquad \dfrac{\partial^{2}\ell}{\partial F^{2}} \;=\; p\,(1-p)\,, \qquad r \;=\; y-p$',
'e2754': r'$F_0 \;=\; \ln\dfrac{\pi}{1-\pi}\,, \qquad \pi \;=\; \dfrac{1}{n}\sum_{i=1}^{n}y_i$',
'e2755': r'$\sum_{i\in t}\ell(y_i,\,F_i+\gamma) \;\approx\; \sum_{i\in t}\ell(y_i,\,F_i)\;-\;\gamma\sum_{i\in t}r_i\;+\;\dfrac{\gamma^{2}}{2}\sum_{i\in t}p_i\,(1-p_i)$',
'e2756': r'$\gamma_t \;=\; \dfrac{\sum_{i\in t}r_i}{\sum_{i\in t}p_i\,(1-p_i)}$',
'e2757': r'$p_{ik} \;=\; \dfrac{e^{F_k(\mathbf{x}_i)}}{\sum_{j=1}^{K}e^{F_j(\mathbf{x}_i)}}\,, \qquad r_{ik} \;=\; \mathbf{1}\,[\,y_i=k\,]-p_{ik}$',
'e2758': r'$\gamma_{tk} \;=\; \dfrac{K-1}{K}\;\dfrac{\sum_{i\in t}r_{ik}}{\sum_{i\in t}p_{ik}\,(1-p_{ik})}$',
'e2759': r'$\ell(y,F) \;=\; e^{-sF}\,, \quad s=2y-1\,, \qquad p \;=\; \sigma(2F)$',
'e27510': r'$w_i \;=\; e^{-s_iF(\mathbf{x}_i)}\,, \qquad r_i \;=\; s_i\,w_i\,, \qquad \gamma_t \;=\; \dfrac{\sum_{i\in t}s_i\,w_i}{\sum_{i\in t}w_i}$',
'e2761': r'$F(\mathbf{x}) \;=\; F_0 \;+\; f_1(x_1)\;+\;f_2(x_2)\;+\;\cdots\;+\;f_p(x_p)$',
'e2771': r'$\gamma_{\mathrm{left}}=\dfrac{4\times(-0.5)}{4\times 0.25}=-2\,, \qquad \gamma_{\mathrm{right}}=\dfrac{5\times 0.5-0.5}{6\times 0.25}=\dfrac{2}{1.5}=1.3333$',
'e27121': r'$g_i \;=\; \dfrac{\partial\ell}{\partial F} \;=\; -r_i\,, \qquad h_i \;=\; \dfrac{\partial^{2}\ell}{\partial F^{2}}\,, \qquad G_t=\sum_{i\in t}g_i\,, \quad H_t=\sum_{i\in t}h_i$',
'e27122': r'$\gamma_t \;=\; -\dfrac{G_t}{H_t}\,, \qquad \sum_{t}\left[\,G_t\,\gamma_t+\frac{1}{2}H_t\,\gamma_t^{2}\right]_{\gamma_t=-G_t/H_t} \;=\; -\dfrac{1}{2}\sum_{t}\dfrac{G_t^{2}}{H_t}$',
'e27123': r'$\mathrm{gain} \;=\; \dfrac{G_L^{2}}{H_L}+\dfrac{G_R^{2}}{H_R}-\dfrac{G^{2}}{H}$',
}
mkeq(E,'.')
print('eqs ok')
