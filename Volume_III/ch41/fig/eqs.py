# Equation images of Chapter 41.  Run: python3 fig/eqs.py
import sys, os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
B = ''
E = {
 'e4131': ['$a(i) \\;=\\; \\dfrac{1}{n_k-1}\\sum_{j\\in C_k,\\;j\\neq i} d(i,j)\\,, \\qquad d(i,j) = \\Vert\\mathbf{x}_i-\\mathbf{x}_j\\Vert$', '$b(i) \\;=\\; \\min_{l\\neq k}\\;\\dfrac{1}{n_l}\\sum_{j\\in C_l} d(i,j)$'],
 'e4132': '$s(i) \\;=\\; \\dfrac{b(i)-a(i)}{\\max\\{a(i),\\,b(i)\\}}$',
 'e4133': ['$\\mathrm{CH} \\;=\\; \\dfrac{\\mathrm{SSB}/(K-1)}{\\mathrm{SSW}/(n-K)}\\,,$', '$\\mathrm{SSB} = \\sum_{k=1}^{K} n_k\\Vert\\boldsymbol{\\mu}_k-\\bar{\\mathbf{x}}\\Vert^{2}\\,, \\qquad \\mathrm{SSW} = \\sum_{k=1}^{K}\\sum_{i\\in C_k}\\Vert\\mathbf{x}_i-\\boldsymbol{\\mu}_k\\Vert^{2}$'],
 'e4134': ['$S_k \\;=\\; \\dfrac{1}{n_k}\\sum_{i\\in C_k}\\Vert\\mathbf{x}_i-\\boldsymbol{\\mu}_k\\Vert\\,, \\qquad \\mathrm{DB} \\;=\\; \\dfrac{1}{K}\\sum_{k=1}^{K}\\max_{l\\neq k}\\;\\dfrac{S_k+S_l}{\\Vert\\boldsymbol{\\mu}_k-\\boldsymbol{\\mu}_l\\Vert}$'],
 'e4141': ['$P_{11} \\;=\\; \\sum_{i,j}\\,\\mathrm{C}(n_{ij},\\,2)\\,, \\qquad P_{11}+P_{10} \\;=\\; \\sum_{i}\\,\\mathrm{C}(n_{i\\cdot},\\,2)\\,,$', '$P_{11}+P_{01} \\;=\\; \\sum_{j}\\,\\mathrm{C}(n_{\\cdot j},\\,2)\\,, \\qquad P_{00} \\;=\\; \\mathrm{C}(n,\\,2)-P_{11}-P_{10}-P_{01}$'],
 'e4142': '$\\mathrm{RI} \\;=\\; \\dfrac{P_{11}+P_{00}}{\\mathrm{C}(n,\\,2)}$',
 'e4143': '$\\mathrm{E}[P_{11}] \\;=\\; \\dfrac{\\left(\\sum_{i}\\,\\mathrm{C}(n_{i\\cdot},\\,2)\\right)\\left(\\sum_{j}\\,\\mathrm{C}(n_{\\cdot j},\\,2)\\right)}{\\mathrm{C}(n,\\,2)}$',
 'e4144': '$\\mathrm{ARI} \\;=\\; \\dfrac{P_{11}-\\mathrm{E}[P_{11}]}{\\dfrac{1}{2}\\left[\\sum_{i}\\,\\mathrm{C}(n_{i\\cdot},\\,2)+\\sum_{j}\\,\\mathrm{C}(n_{\\cdot j},\\,2)\\right]-\\mathrm{E}[P_{11}]}$',
 'e4145': ['$H(c) \\;=\\; -\\sum_{i}\\dfrac{n_{i\\cdot}}{n}\\ln\\dfrac{n_{i\\cdot}}{n}\\,, \\qquad H(u) \\;=\\; -\\sum_{j}\\dfrac{n_{\\cdot j}}{n}\\ln\\dfrac{n_{\\cdot j}}{n}$'],
 'e4146': '$\\mathrm{MI} \\;=\\; \\sum_{i,j}\\dfrac{n_{ij}}{n}\\ln\\dfrac{n\\,n_{ij}}{n_{i\\cdot}\\,n_{\\cdot j}}$',
 'e4147': ['$h \\;=\\; \\dfrac{\\mathrm{MI}}{H(c)} \\;=\\; 1-\\dfrac{H(c\\,|\\,u)}{H(c)}\\,, \\qquad \\kappa \\;=\\; \\dfrac{\\mathrm{MI}}{H(u)} \\;=\\; 1-\\dfrac{H(u\\,|\\,c)}{H(u)}\\,,$', '$V \\;=\\; \\dfrac{2\\,h\\,\\kappa}{h+\\kappa}$'],
}
import matplotlib.pyplot as plt
def mk2(name, lines, step=0.62):
    n = len(lines); fig = plt.figure(figsize=(0.01, n * step))
    for j, t in enumerate(lines): fig.text(0, ((n - 1 - j) * step + 0.12) / (n * step), t, fontsize=11.5)
    fig.savefig(name + '.png', dpi=300, bbox_inches='tight', pad_inches=0.02, facecolor='white'); plt.close(fig)
multi = {k: v for k, v in E.items() if isinstance(v, list)}
mkeq({k: v for k, v in E.items() if k not in multi}, '.')
for k, v in multi.items(): mk2(k, v, 0.62 if k in ('e4131', 'e4141') else 0.5)
import glob
from PIL import Image
w = {os.path.basename(f): Image.open(f).size[0] / 300 for f in glob.glob('e41*.png')}
print('eqs ok', len(E), 'max width in', round(max(w.values()), 2), [k for k, v in w.items() if v > 5.2])
