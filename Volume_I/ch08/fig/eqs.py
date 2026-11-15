# Equation images of Chapter 8. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq
E={
'e811': r'$n_{\mathrm{test}} \;=\; \lceil f\,n\rceil\,, \qquad n_{\mathrm{train}} \;=\; n - n_{\mathrm{test}}$',
'e821': r'$\hat{y}(\mathbf{x}) \;=\; \bar{y}_{\mathrm{train}} \;=\; \dfrac{1}{n}\sum_{i=1}^{n} y_i \qquad \mathrm{for\ every\ }\mathbf{x}$',
'e831': r'$r_i \;=\; y_i - \hat{y}_i$',
'e832': r'$\mathrm{MSE} \;=\; \dfrac{1}{n}\sum_{i=1}^{n} r_i^{\,2}\,, \qquad \mathrm{RMSE} \;=\; \sqrt{\mathrm{MSE}}\,, \qquad \mathrm{MAE} \;=\; \dfrac{1}{n}\sum_{i=1}^{n} \vert r_i\vert$',
'e833': r'$R^2 \;=\; 1-\dfrac{\sum_i (y_i-\hat{y}_i)^2}{\sum_i (y_i-\bar{y})^2} \;=\; 1-\dfrac{\mathrm{MSE\ of\ the\ model}}{\mathrm{MSE\ of\ the\ constant\ }\bar{y}}$',
'e841': r'$\mathrm{accuracy} \;=\; \dfrac{\mathrm{TP}+\mathrm{TN}}{n}$',
'e842': r'$\mathrm{precision} \;=\; \dfrac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}\,, \qquad \mathrm{recall} \;=\; \dfrac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}$',
'e843': r'$F_1 \;=\; \dfrac{2PR}{P+R} \;=\; \dfrac{2\,\mathrm{TP}}{2\,\mathrm{TP}+\mathrm{FP}+\mathrm{FN}}$',
'e851': r'$\bar{s} \;=\; \dfrac{1}{k}\sum_{j=1}^{k} s_j\,, \qquad s_{\mathrm{sd}} \;=\; \sqrt{\dfrac{1}{k-1}\sum_{j=1}^{k}(s_j-\bar{s})^2}$',
'e861': r'$\mathrm{scorer\ value} \;=\; -\,\mathrm{MSE}$',
}
mkeq(E,'.'); print('eqs ok')
