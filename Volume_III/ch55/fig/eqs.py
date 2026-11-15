# Equation images of Chapter 55. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq
E={
'e5531': [r"$\mathrm{CV}\;=\;\dfrac{1}{K}\sum_{k=1}^{K}s_k\,,\qquad s_k\;=\;\mathrm{score}\,\left(f_{-k};\,D_k\right)$"],
'e5532': [r"$a_{kc}\;=\;\#\left\{\,j:\ s_c\leq j<s_c+n_c\,,\ \ j\equiv k-1\ \ (\mathrm{mod}\ K)\right\}\,,\qquad s_c=n_1+\cdots+n_{c-1}$"],
'e5533': [r"$y_{ij}=a_i+e_{ij}\,,\quad a_i\sim N(0,\sigma_a^{2})\,,\ \ e_{ij}\sim N(0,\sigma^{2})$",
          r"$\mathrm{MSE}_{\mathrm{same}}\;=\;\sigma^{2}\left(1+\dfrac{1}{m-1}\right)\,,\qquad \mathrm{MSE}_{\mathrm{new}}\;\approx\;\sigma_a^{2}+\sigma^{2}$"],
'e5534': [r"$\mathrm{Var}(\overline{s})\;=\;\dfrac{\sigma^{2}}{K}\left[\,1+(K-1)\,\rho\,\right]\,,\qquad \dfrac{\mathrm{true\ s.e.}}{\mathrm{naive\ s.e.}}\;\approx\;\sqrt{\dfrac{1+(K-1)\rho}{1-\rho}}$"],
'e5535': [r"$t_k\;=\;n-(K-k+1)\,h\,,\qquad h=\lfloor n/(K+1)\rfloor$",
          r"$\mathrm{train}_k=\{0,\dots,t_k-g-1\}\,,\qquad \mathrm{test}_k=\{t_k,\dots,t_k+h-1\}$"],
'e5536_unused': [r"$\mathrm{CV}_{\mathrm{nested}}\;=\;\dfrac{1}{K_o}\sum_{k=1}^{K_o}\mathrm{score}\,\left(f_{\hat{\theta}_k};\,D_k\right)\,,\qquad \hat{\theta}_k\;=\;\arg\max_{\theta}\ \mathrm{CV}_{\mathrm{inner}}\,\left(\theta;\ D\setminus D_k\right)$"],
}
mkeq(E,'.')
