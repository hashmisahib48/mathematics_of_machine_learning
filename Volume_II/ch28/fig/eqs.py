# Equation images of Chapter 28. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq
E={
'e2831': r'$t_{jk} \;=\; \mathrm{quantile}_{\,k/B}\!\left(x_{1j},\dots,x_{nj}\right),\quad k=1,\dots,B-1$',
'e2832': r'$b_{ij} \;=\; \#\{\,k:\ t_{jk} < x_{ij}\,\}\ \in\ \{0,\dots,B-1\}\,, \qquad b_{ij}=B_{\max}\ \ \mathrm{if}\ x_{ij}\ \mathrm{is\ missing}$',
'e2833': r'$F(\mathbf{x}) \;=\; F_0 \;+\; \eta\sum_{m=1}^{M} f_m(\mathbf{x})\,, \qquad f_m(\mathbf{x}) \;=\; w_{t(\mathbf{x})}$',
'e2834': r'$g_i \;=\; \dfrac{\partial\,\ell(y_i,F_i)}{\partial F_i}\,, \qquad h_i \;=\; \dfrac{\partial^{2}\,\ell(y_i,F_i)}{\partial F_i^{\,2}}\,, \qquad F_i \;=\; F_{m-1}(\mathbf{x}_i)$',
'e2835': r'$J(\mathbf{w}) \;\approx\; \sum_{t=1}^{T}\left[\,G_t\,w_t \;+\; \dfrac{1}{2}\,(H_t+\lambda)\,w_t^{2}\right], \qquad G_t=\sum_{i\in t} g_i\,,\ \ H_t=\sum_{i\in t} h_i$',
'e2836': r'$w_t^{*} \;=\; -\,\dfrac{G_t}{H_t+\lambda}\,, \qquad J_t^{*} \;=\; -\,\dfrac{1}{2}\,\dfrac{G_t^{2}}{H_t+\lambda}$',
'e2837': r'$G_{jb} \;=\; \sum_{i:\ b_{ij}=b} g_i\,, \qquad H_{jb} \;=\; \sum_{i:\ b_{ij}=b} h_i\,, \qquad n_{jb} \;=\; \#\{\,i:\ b_{ij}=b\,\}$',
'e2838': r'$(G_L,H_L,n_L) \;=\; \sum_{b\,\leq\,k}\,(G_{jb},H_{jb},n_{jb})\,, \qquad (G_R,H_R,n_R) \;=\; (G,H,n)-(G_L,H_L,n_L)$',
'e2839': r'$\mathrm{gain}(j,k) \;=\; \dfrac{G_L^{2}}{H_L+\lambda} \;+\; \dfrac{G_R^{2}}{H_R+\lambda} \;-\; \dfrac{G^{2}}{H+\lambda}$',
'e28310': r'$F_m(\mathbf{x}) \;=\; F_{m-1}(\mathbf{x}) \;+\; \eta\,w_{t(\mathbf{x})}\,, \qquad F_0 \;=\; \arg\min_{c}\ \sum_{i}\ \ell(y_i,c)$',
'e28311': r'$(G_b,\ H_b,\ n_b)_{\mathrm{larger}} \;=\; (G_b,\ H_b,\ n_b)_{\mathrm{parent}} \;-\; (G_b,\ H_b,\ n_b)_{\mathrm{smaller}}\,, \qquad b=0,1,\dots$',
'e28312': r'$\kappa_c \;=\; \dfrac{G_c}{H_c+10}\,, \qquad \mathrm{sort\ the\ categories\ by\ }\kappa_c\,,\ \ \mathrm{then\ scan\ it\ like\ the\ bins}$',
'e2851': r'$\mathrm{gain}(0) \;=\; \dfrac{1.25^{2}}{0.7292}+\dfrac{(-1.25)^{2}}{2.1875} \;=\; 2.1429+0.7143 \;=\; 2.8571$',
'e2852': r'$w_L \;=\; -\dfrac{1.25}{0.7292} \;=\; -1.7143\,, \qquad w_R \;=\; -\dfrac{-1.25}{2.1875} \;=\; 0.5714$',
'e2853': r'$\mathrm{gain}(0) \;=\; \dfrac{1.25^{2}}{0.7292+1}+\dfrac{(-1.25)^{2}}{2.1875+1} \;=\; 0.9036+0.4902 \;=\; 1.3938$',
}
mkeq(E,'.')
print('eqs ok')
