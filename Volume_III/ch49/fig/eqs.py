# Equation images of Chapter 49. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq
E={
'e4931': r"$\tilde{x}\;=\;\dfrac{x-c}{s}\,,\qquad c=\mathrm{centre}\ (\mathrm{shift})\,,\qquad s=\mathrm{scale}\ (s>0)$",
'e4932': r"$d^{2}_{\mathrm{scaled}}(\mathbf{x},\mathbf{z})\;=\;\sum_{j=1}^{p}\dfrac{(x_j-z_j)^{2}}{s_j^{2}}\,,\qquad \mathrm{E}\,d^{2}_{\mathrm{scaled}}\;=\;2\sum_{j=1}^{p}\dfrac{\sigma_j^{2}}{s_j^{2}}\;=\;2p\quad(s_j=\sigma_j)$",
'e4933': r"$\alpha\sum_{j=1}^{p}w_j^{2}\;=\;\alpha\sum_{j=1}^{p}\dfrac{\tilde{w}_j^{2}}{s_j^{2}}\,,\qquad \tilde{w}_j=s_j\,w_j$",
'e4934': r"$\tilde{\mathbf{x}}_i\;=\;\dfrac{\mathbf{x}_i}{\Vert\mathbf{x}_i\Vert}\,,\qquad \Vert\mathbf{x}\Vert_1=\sum_{j}\,|x_j|\,,\quad \Vert\mathbf{x}\Vert_2=\sqrt{\sum_{j}\,x_j^{2}}\,,\quad \Vert\mathbf{x}\Vert_{\max}=\max_{j}|x_j|$",
'e4935': [r"$\psi(x;\lambda)\;=\;\dfrac{x^{\lambda}-1}{\lambda}\,,\quad \lambda\neq 0\,,\qquad\quad \psi(x;0)\;=\;\ln x\,,\qquad x>0$"],
'e4936': [r"$\psi(x;\lambda)=[(x+1)^{\lambda}-1]/\lambda\,,\ \ x\geq 0,\ \lambda\neq 0\,;\qquad \ln(x+1)\,,\ \ x\geq 0,\ \lambda=0$",
          r"$\psi(x;\lambda)=-[(1-x)^{2-\lambda}-1]/(2-\lambda)\,,\ \ x<0,\ \lambda\neq 2\,;\qquad -\ln(1-x)\,,\ \ x<0,\ \lambda=2$"],
'e4937': [r"$\ell(\lambda)\;=\;-(n/2)\,\ln\hat{\sigma}^{2}(\lambda)\;+\;(\lambda-1)\sum_{i=1}^{n}\ln x_i\,,$",
          r"$\hat{\sigma}^{2}(\lambda)\;=\;(1/n)\,\sum_{i=1}^{n}\left(\psi(x_i;\lambda)-\bar{\psi}\right)^{2}$"],
'e4938': [r"$u(x)\;=\;r_k+\dfrac{x-q_k}{q_{k+1}-q_k}\,(r_{k+1}-r_k)\,,\quad q_k\leq x\leq q_{k+1}\,,\qquad r_k=\dfrac{k}{m-1}$",
          r"$\tilde{x}\;=\;u(x)\ \ (\mathrm{uniform})\qquad\mathrm{or}\qquad \tilde{x}\;=\;\Phi^{-1}\!\left(u(x)\right)\ \ (\mathrm{normal})$"],
}
mkeq(E,'.')
