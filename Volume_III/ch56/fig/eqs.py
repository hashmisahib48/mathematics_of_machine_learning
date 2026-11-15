# Equation images of Chapter 56. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq
E={
'e5631': [r"$\hat{\theta}\;=\;\arg\max_{\theta\in\Theta}\ \mathrm{CV}(\theta)\,,\qquad \mathrm{CV}(\theta)\;=\;\dfrac{1}{K}\sum_{k=1}^{K}s_k(\theta)$"],
'e5632': [r"$|\Theta|\;=\;\sum_{g}\ \prod_{j}\,m_{gj}\,,\qquad \mathrm{fits}\;=\;K\,|\Theta|+1$"],
'e5633': [r"$P(\mathrm{hit})\;=\;1-(1-q)^{n}\,,\qquad n\;\geq\;\dfrac{\ln(1-P)}{\ln(1-q)}$"],
'e5634': [r"$n_k\;=\;\lceil\, n_0/\eta^{k}\,\rceil\,,\qquad r_k\;=\;\min\left(\eta^{k}\,r_0\,,\ R\right)\,,\qquad k=0,1,\dots$"],
'e5635': [r"$n_{\mathrm{req}}=1+\lfloor\log_{\eta}n_0\rfloor\,,\qquad n_{\mathrm{poss}}=1+\lfloor\log_{\eta}\lfloor R/r_0\rfloor\rfloor$",
          r"$n_{\mathrm{rungs}}\;=\;\min\left(n_{\mathrm{req}}\,,\ n_{\mathrm{poss}}\right)$"],
'e5636': [r"$\max_j\,\mu_j\;\leq\;E\left[\,\max_j\,\hat{J}_j\,\right]\;\leq\;\max_j\,\mu_j+\sigma\sqrt{2\ln M}\,,\qquad \hat{J}_j=\mu_j+\varepsilon_j$"],
'e5637': [r"$\mathrm{CV}_{\mathrm{nested}}\;=\;\dfrac{1}{K_o}\sum_{k=1}^{K_o}\mathrm{score}\left(f_{\hat{\theta}_k};\,D_k\right)$",
          r"$\hat{\theta}_k\;=\;\arg\max_{\theta}\ \mathrm{CV}_{\mathrm{inner}}\left(\theta;\ D\setminus D_k\right)$"],
}
mkeq(E,'.')
