# Equation images of Chapter 7. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq
import matplotlib.pyplot as plt
X=r'\mathbf{X}'; y=r'\mathbf{y}'; w=r'\hat{\mathbf{w}}'
E={
'e711': rf'$({w},\,\hat{{b}}) \;=\; \mathrm{{fit}}({X},\,{y};\;\mathbf{{h}})\,, \qquad \hat{{y}}_{{\mathrm{{new}}}} \;=\; f(\mathbf{{x}}_{{\mathrm{{new}}}};\;{w},\,\hat{{b}})$',
'e721': r'$x_{ij} \;=\; \mathrm{data}[k]\ \ \mathrm{with}\ \ j=\mathrm{indices}[k]\,, \qquad \mathrm{indptr}[i]\;\leq\; k \;<\; \mathrm{indptr}[i+1]$',
'e751': r'$P_k \;=\; \dfrac{e^{f_k}}{\sum_{j=1}^{K} e^{f_j}}\,, \qquad k = 1,\ldots,K$',
'e752': r'$P_k \;=\; \dfrac{e^{\,f_k - f_{\max}}}{\sum_{j=1}^{K} e^{\,f_j - f_{\max}}}$',
'e762': r'$z_{ij} \;=\; \dfrac{x_{ij}-\mu_j}{\sigma_j}$',
'e763': r'$x_{ij} \;=\; \sigma_j\,z_{ij} + \mu_j$',
'e771': r'$\hat{y} \;=\; f_{\mathrm{clf}}\left(T_{\mathrm{scale}}(\mathbf{x})\right) \;=\; f_{\mathrm{clf}}\!\left(\dfrac{\mathbf{x}-\mu}{\sigma}\right)$',
'e791': r'$s \;\approx\; \sqrt{\dfrac{a\,(1-a)}{m}}$',
}
mkeq(E,'.')
fig=plt.figure(figsize=(.01,.01)); fig.text(0,0,r'$\mu_j \;=\; \dfrac{1}{n}\sum_{i=1}^{n} x_{ij}\,, \qquad \sigma_j \;=\; \sqrt{\dfrac{1}{n}\sum_{i=1}^{n}\,(x_{ij}-\mu_j)^2}$',fontsize=11.5); fig.savefig('e761.png',dpi=300,bbox_inches='tight',pad_inches=0.02,facecolor='white'); plt.close(fig)
print('eqs ok')
