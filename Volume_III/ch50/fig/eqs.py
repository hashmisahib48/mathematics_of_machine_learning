# Equation images of Chapter 50. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq
E={
# 50.3 one-hot and ordinal
'e5032': r'$\mathbf{D}^{T}\mathbf{D}\;=\;\mathrm{diag}(n_1,\ldots,n_K)\,,\qquad \mathbf{D}^{T}\mathbf{y}\;=\;(n_1\bar{y}_1,\ldots,n_K\bar{y}_K)^{T}\;\Rightarrow\;w_k\;=\;\bar{y}_k$',
'e5033': r'$w_k\;=\;\dfrac{n_k\,(\bar{y}_k-\bar{y})}{n_k+\alpha}\,,\qquad \hat{y}\;=\;\bar{y}+w_k\;=\;\dfrac{n_k\,\bar{y}_k+\alpha\,\bar{y}}{n_k+\alpha}$',
# 50.4 target encoding
'e5041': [r'$t_c\;=\;\arg\min_{t}\;\left[\sum_{i\in c}(y_i-t)^{2}+m\,(\bar{y}-t)^{2}\right]\;=\;\dfrac{n_c\,\bar{y}_c+m\,\bar{y}}{n_c+m}$',
          r'$=\;\lambda_c\,\bar{y}_c+(1-\lambda_c)\,\bar{y}\,,\qquad \lambda_c\;=\;\dfrac{n_c}{n_c+m}$'],
'e5043': [r'$\lambda_c\;=\;\dfrac{n_c/\sigma_c^{2}}{n_c/\sigma_c^{2}+1/\tau^{2}}\;=\;\dfrac{n_c}{n_c+m_c}\,,\qquad m_c\;=\;\dfrac{\sigma_c^{2}}{\tau^{2}}$',
          r'$\sigma_c^{2}=\dfrac{1}{n_c}\sum_{i\in c}(y_i-\bar{y}_c)^{2}\,,\qquad \tau^{2}=\dfrac{1}{n}\sum_{i=1}^{n}(y_i-\bar{y})^{2}$'],
# 50.5 binning
'e5051': r'$\mathrm{uniform:}\ b_j=x_{\min}+j\,\dfrac{x_{\max}-x_{\min}}{B}\,,\qquad \mathrm{quantile:}\ b_j=\mathrm{percentile}\ \dfrac{100\,j}{B}\ \mathrm{of}\ x$',
}
mkeq(E,'.')
