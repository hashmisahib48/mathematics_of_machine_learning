# Equation images of Chapter 54. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq
E={
# 54.3 pipeline
'e5431': [r'$\theta_j=\mathrm{fit}_j(\mathbf{X}_{j-1},\mathbf{y})\,,\qquad \mathbf{X}_j=T_j(\mathbf{X}_{j-1};\theta_j)\,,\qquad j=1,\ldots,m-1\,,\qquad \mathbf{X}_0=\mathbf{X}$',
          r'$\theta_m=\mathrm{fit}_m(\mathbf{X}_{m-1},\mathbf{y})\,,\qquad \hat{y}(\mathbf{x})=f_m\left(T_{m-1}(\cdots T_2(T_1(\mathbf{x};\theta_1);\theta_2)\cdots;\theta_{m-1});\theta_m\right)$'],
'e5432': [r'$\hat{E}_{\mathrm{honest}}=\frac{1}{K}\sum_{k}L_k\left(\theta^{(-k)}\right)$',
          r'$\hat{E}_{\mathrm{leaky}}=\frac{1}{K}\sum_{k}L_k\left(\theta^{(\mathrm{all})}_{\mathrm{steps}},\,\theta^{(-k)}_{\mathrm{model}}\right)$'],
'e5433': r'$\mathrm{E}\,\max_{j\leq p}\,|r_j|\;\approx\;\sqrt{\dfrac{2\,\ln(2p)}{n}}$',
# 54.4 routing
'e5441': [r'$\mathrm{ColumnTransformer:}\quad \mathbf{z}=\left(T_1(\mathbf{x}_{S_1}),\;T_2(\mathbf{x}_{S_2}),\;\ldots,\;T_r(\mathbf{x}_{S_r}),\;\mathbf{x}_{R}\right)$',
          r'$\mathrm{FeatureUnion:}\qquad\quad \mathbf{z}=\left(w_1\,T_1(\mathbf{x}),\;w_2\,T_2(\mathbf{x}),\;\ldots,\;w_r\,T_r(\mathbf{x})\right)$'],
# 54.5 target transform
'e5451': [r'$\ln y=\mu(\mathbf{x})+\varepsilon\,,\qquad \varepsilon\sim N(0,\sigma^{2})\,,\qquad \mathrm{median}(y\,|\,\mathbf{x})=e^{\mu(\mathbf{x})}$',
          r'$\mathrm{E}(y\,|\,\mathbf{x})=e^{\mu(\mathbf{x})}\,\mathrm{E}\,e^{\varepsilon}=e^{\mu(\mathbf{x})}\,e^{\sigma^{2}/2}$'],
'e5452': r'$\hat{y}_{\mathrm{smear}}(\mathbf{x})=e^{\hat{\mu}(\mathbf{x})}\cdot\dfrac{1}{n}\sum_{i=1}^{n}e^{\hat{\varepsilon}_i}\,,\qquad \hat{\varepsilon}_i=\ln y_i-\hat{\mu}(\mathbf{x}_i)$',
}
mkeq(E,'.')
