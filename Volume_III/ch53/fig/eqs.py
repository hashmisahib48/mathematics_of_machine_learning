# Equation images of Chapter 53. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq
E={
'e5331': [r"$\mathrm{Var}(x_j)\;=\;\dfrac{1}{n}\sum_{i=1}^{n}\left(x_{ij}-\bar{x}_j\right)^{2}$", r"$\mathrm{Var}(x_j)\;=\;q_j\,(1-q_j)\qquad(x_{ij}\in\{0,1\},\ q_j=\bar{x}_j)$"],
'e5332': [r"$F_j\;=\;\dfrac{\sum_{k=1}^{K} n_k\,(\bar{x}_{kj}-\bar{x}_j)^{2}\,/\,(K-1)}{\sum_{k=1}^{K}\sum_{i\in C_k}(x_{ij}-\bar{x}_{kj})^{2}\,/\,(n-K)}\;=\;\dfrac{\mathrm{SSB}_j\,/\,(K-1)}{\mathrm{SSW}_j\,/\,(n-K)}$"],
'e5333': [r"$F_j\;=\;\dfrac{r_j^{2}}{1-r_j^{2}}\,(n-2)\,,\qquad r_j\;=\;\dfrac{\sum_{i}(x_{ij}-\bar{x}_j)(y_i-\bar{y})}{\sqrt{\sum_{i}(x_{ij}-\bar{x}_j)^{2}\,\sum_{i}(y_i-\bar{y})^{2}}}$"],
'e5334': [r"$\chi^{2}_j\;=\;\sum_{k=1}^{K}\dfrac{(O_{kj}-E_{kj})^{2}}{E_{kj}}\,,\qquad O_{kj}=\sum_{i\in C_k}x_{ij}\,,\qquad E_{kj}=\dfrac{n_k}{n}\sum_{i=1}^{n}x_{ij}$"],
'e5335': [r"$I(x_j;y)\;=\;\sum_{k}\int p(x,k)\,\ln\dfrac{p(x,k)}{p(x)\,P(k)}\,dx$", r"$\hat{I}\;=\;\psi(n)+\overline{\psi(\kappa_i)}-\overline{\psi(n_{y_i})}-\overline{\psi(m_i)}$"],
'e5336': [r"$P(\mathrm{at\ least\ one\ false\ selection})\;=\;1-(1-\alpha)^{m}\;\approx\;m\,\alpha$", r"$(m\ \mathrm{unrelated\ features,\ each\ tested\ at\ level}\ \alpha)$"],
'e5337': [r"$\mathrm{select}\ j\ \Longleftrightarrow\ p_j<\dfrac{\alpha}{m}\qquad(\mathrm{family\!-\!wise\ error\ rate}\ \leq\alpha)$"],
'e5338': [r"$p_{(1)}\leq p_{(2)}\leq\cdots\leq p_{(m)}\,,\qquad k^{*}=\max\left\{k:\ p_{(k)}\leq\dfrac{k}{m}\,\alpha\right\}$", r"$\mathrm{select}\ j\ \Longleftrightarrow\ p_j\leq p_{(k^{*})}$"],
'e5341': [r"$c_j\;=\;w_j^{2}\ \ (\mathrm{linear\ models})\,,\qquad c_j=\mathrm{importance}_j\ \ (\mathrm{trees})$", r"$S\leftarrow S\setminus\{\mathrm{the\ }\min(\mathrm{step},\,|S|-s)\ \mathrm{smallest}\ c_j\}$"],
'e5342': [r"$\min_{\mathbf{w},\,b}\;\;\|\mathbf{w}\|_{1}\;+\;C\sum_{i=1}^{n}\ell_i(\mathbf{w},b)\,,\qquad \mathrm{keep}\ j\ \Longleftrightarrow\ \sum_{k}|w_{kj}|\;\geq\;t$"],
'e5343': [r"$N_{\mathrm{fits}}\;=\;N_{\mathrm{cv}}\sum_{t=0}^{s-1}(p-t)\;=\;N_{\mathrm{cv}}\left(sp-\dfrac{s(s-1)}{2}\right)$"],
'e5351': [r"$\mathrm{E}\,\max_{j\leq m}\,|r_j|\;\approx\;\sqrt{\dfrac{2\ln m}{n}}\qquad(\mathrm{all}\ m\ \mathrm{features\ unrelated\ to}\ y)$"],
}
mkeq(E,'.')
