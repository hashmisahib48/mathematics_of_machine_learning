# Equation images of Chapter 52. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq
E={
'e5231': [r"$n_{ij}\;=\;\#\{\,t:\ \mathrm{token}_t(d_i)=w_j\,\}\,,\qquad \mathbf{X}=[\,n_{ij}\,]\in\mathbb{R}^{n\times V}$",
          r"$\mathrm{df}_j\;=\;\#\{\,i:\ n_{ij}>0\,\}\,,\qquad \mathrm{min\_df}\ \leq\ \mathrm{df}_j\ \leq\ \mathrm{max\_df}$"],
'e5232': r"$N_{\mathrm{grams}}\;=\;\sum_{k=n_{\min}}^{n_{\max}}\max\,(T-k+1,\,0)\,,\qquad T=\mathrm{number\ of\ tokens}$",
'e5241': r"$w_{ij}\;=\;t_{ij}\,\mathrm{idf}_j\,,\qquad t_{ij}=n_{ij}\ \ \mathrm{or,\ with\ sublinear\_tf,}\ \ t_{ij}=1+\ln n_{ij}\ \ (n_{ij}>0)$",
'e5242': r"$\mathrm{idf}_j\;=\;\ln\dfrac{1+n}{1+\mathrm{df}_j}\;+\;1\,,\qquad \mathrm{idf}_j\;=\;\ln\dfrac{n}{\mathrm{df}_j}\;+\;1\ \ (\mathrm{smooth\_idf=False})$",
'e5243': [r"$\mathbf{z}_i\;=\;\dfrac{\mathbf{w}_i}{\Vert\mathbf{w}_i\Vert_2}\,,\qquad \cos(\mathbf{z}_i,\mathbf{z}_k)\;=\;\mathbf{z}_i^{T}\mathbf{z}_k\,,$",
          r"$ \Vert\mathbf{z}_i-\mathbf{z}_k\Vert^{2}\;=\;2\,\left(1-\mathbf{z}_i^{T}\mathbf{z}_k\right)$"],
'e5251': [r"$b_j\;=\;|h(w_j)|\ \mathrm{mod}\ m\,,\qquad \xi_j\;=\;+1\ (h\geq 0)\ \ \mathrm{or}\ \ -1\ (h<0)$",
          r"$\phi_b(\mathbf{x})\;=\;\sum_{j:\ b_j=b}\xi_j\,x_j\,,\qquad b=0,\dots,m-1$"],
'e5252': r"$\phi(\mathbf{x})^{T}\phi(\mathbf{z})\;=\;\sum_{j}\,x_j z_j\;+\;\sum_{j\neq k,\ b_j=b_k}\xi_j\xi_k\,x_j z_k$",
'e5253': r"$\mathrm{E}\,[\,\#\ \mathrm{occupied\ buckets}\,]\;=\;m\,\left[\,1-(1-1/m)^{V}\,\right]$",
}
mkeq(E,'.')
