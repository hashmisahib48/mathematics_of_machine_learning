# Equation images of Chapter 6. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq
T=r'^{\mathrm{T}}'; X=r'\mathbf{X}'; y=r'\mathbf{y}'; w=r'\mathbf{w}'; I=r'\mathbf{I}'; H=r'\mathbf{H}'; ep=r'\mathbf{\varepsilon}'
E={
'e611': r'$J_{\mathrm{train}} \;=\; \dfrac{1}{n}\sum_{i=1}^{n}\,\left(y_i-\hat{f}(\mathbf{x}_i)\right)^2$',
'e612': r"$J_{\mathrm{test}} \;=\; \dfrac{1}{m}\sum_{j=1}^{m}\,\left(\tilde{y}_j-\hat{f}(\tilde{\mathbf{x}}_j)\right)^2$",
'e613': r'$\mathrm{gap} \;=\; J_{\mathrm{test}} - J_{\mathrm{train}}$',
'e614': r'$\mathrm{Err}(\hat{f}) \;=\; E\!\left[\left(y-\hat{f}(\mathbf{x})\right)^2\right]$',
'e621': r'$\hat{y} \;=\; w_0 + w_1\,x + w_2\,x^2 + \cdots + w_d\,x^d$',
'e631': r'$y \;=\; f(x) + \varepsilon\,, \qquad E[\varepsilon]=0\,, \quad \mathrm{Var}[\varepsilon]=\sigma^2$',
'e632': r'$E\!\left[(y-\hat{f})^2\right] \;=\; E[\varepsilon^2] + 2\,E[\varepsilon]\,E[f-\hat{f}] + E\!\left[(f-\hat{f})^2\right] \;=\; \sigma^2 + E\!\left[(f-\hat{f})^2\right]$',
'e633': r'$E\!\left[(f-\hat{f})^2\right] \;=\; \left(f-E[\hat{f}]\right)^2 \;+\; E\!\left[\left(\hat{f}-E[\hat{f}]\right)^2\right]$',
'e634': r'$E\!\left[(y-\hat{f})^2\right] \;=\; \sigma^2 \;+\; \left(E[\hat{f}]-f\right)^2 \;+\; E\!\left[\left(\hat{f}-E[\hat{f}]\right)^2\right]$',
'e641': rf'$\hat{{{y}}} \;=\; {H}\,{y}\,, \qquad {H} \;=\; {X}\,({X}{T}{X})^{{-1}}{X}{T}$',
'e642': rf'$E\!\left[J_{{\mathrm{{train}}}}\right] \;=\; \dfrac{{1}}{{n}}\,E\!\left[{ep}{T}({I}-{H}){ep}\right] \;=\; \dfrac{{\sigma^2}}{{n}}\,\mathrm{{tr}}({I}-{H}) \;=\; \sigma^2\left(1-\dfrac{{p}}{{n}}\right)$',
'e643': rf'$E\!\left[J_{{\mathrm{{test}}}}\right] \;=\; \dfrac{{1}}{{n}}\left(n\sigma^2 + \sigma^2\,\mathrm{{tr}}({H})\right) \;=\; \sigma^2\left(1+\dfrac{{p}}{{n}}\right)$',
'e644': r'$E\!\left[J_{\mathrm{test}}\right]-E\!\left[J_{\mathrm{train}}\right] \;=\; \dfrac{2\,p\,\sigma^2}{n}$',
'e651': rf'$J_{{\alpha}}({w}) \;=\; \Vert {y}-{X}{w}\Vert^2 \;+\; \alpha\,\Vert {w}\Vert^2$',
'e652': rf'$\hat{{{w}}}_{{\alpha}} \;=\; ({X}{T}{X} + \alpha\,{I})^{{-1}}{X}{T}{y}$',
'e653': r'$w_{\alpha} \;=\; \dfrac{S_{xy}}{S_{xx}+\alpha} \;=\; \dfrac{1}{1+\alpha/S_{xx}}\;\dfrac{S_{xy}}{S_{xx}}$',
}
mkeq(E,'.'); print('eqs ok')
