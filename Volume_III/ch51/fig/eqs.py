# Equation images of Chapter 51. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq
E={
# 51.3 mechanisms and mean fill
'e5131': [r'$\mathrm{MCAR:}\quad P(\mathbf{R}\mid\mathbf{X})\;=\;P(\mathbf{R})$',
          r'$\mathrm{MAR:}\quad\ \ P(\mathbf{R}\mid\mathbf{X})\;=\;P(\mathbf{R}\mid\mathbf{X}_{\mathrm{obs}})$',
          r'$\mathrm{MNAR:}\quad P(\mathbf{R}\mid\mathbf{X})\ \mathrm{depends\ on}\ \mathbf{X}_{\mathrm{mis}}\ \mathrm{even\ given}\ \mathbf{X}_{\mathrm{obs}}$'],
'e5132': [r'$\mathrm{Var}(\tilde{x})\;=\;\dfrac{n_o}{n}\,s_o^{2}\;=\;(1-q)\,s_o^{2}\,,\qquad \mathrm{Cov}(\tilde{x},z)\;\approx\;(1-q)\,s_{xz}$',
          r'$\mathrm{Corr}(\tilde{x},z)\;\approx\;\dfrac{(1-q)\,s_{xz}}{\sqrt{(1-q)\,s_o^{2}\,s_z^{2}}}\;=\;\sqrt{1-q}\;\,r_{xz}$'],
# 51.4 nearest neighbours
'e5141': r'$d(\mathbf{x},\mathbf{z})\;=\;\sqrt{\dfrac{p}{|S|}\sum_{j\in S}\,(x_j-z_j)^{2}}\,,\qquad S=\{j:\ x_j\ \mathrm{and}\ z_j\ \mathrm{both\ observed}\}$',
'e5142': r'$\hat{x}_{ij}\;=\;\dfrac{\sum_{l\in N_k(i,j)}w_l\,x_{lj}}{\sum_{l\in N_k(i,j)}w_l}\,,\qquad w_l=1\ \ (\mathrm{uniform})\ \ \mathrm{or}\ \ w_l=\dfrac{1}{d(\mathbf{x}_i,\mathbf{x}_l)}\ \ (\mathrm{distance})$',
# 51.5 chained regressions
'e5151': [r'$\hat{x}_{ij}\;\leftarrow\;f_j(\mathbf{x}_{i,-j})\,,\qquad f_j\ \mathrm{fitted\ to}\ \{(\mathbf{x}_{l,-j},\,x_{lj}):\ x_{lj}\ \mathrm{observed}\}$',
          r'$x_{ij}\;\sim\;N\!\left(\mu_j(\mathbf{x}_{i,-j}),\ \sigma_j^{2}(\mathbf{x}_{i,-j})\right)\qquad (\mathrm{sample\_posterior=True})$'],
}
mkeq(E,'.')
