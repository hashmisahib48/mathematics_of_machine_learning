# Equation and matrix images of Chapter 9. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq, matimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
T=r'^{\mathrm{T}}'; X=r'\mathbf{X}'; y=r'\mathbf{y}'; w=r'\mathbf{w}'; wh=r'\hat{\mathbf{w}}'
E={
'e931': r'$\hat{y} \;=\; b + w\,x$',
'e932': r'$r_i \;=\; y_i-\hat{y}_i \;=\; y_i - b - w\,x_i$',
'e933': r'$J(w,b) \;=\; \sum_{i=1}^{n} r_i^{\,2} \;=\; \sum_{i=1}^{n}\,(y_i - b - w\,x_i)^2$',
'e934x': [r'$\dfrac{\partial J}{\partial b} \;=\; -2\sum_{i=1}^{n}\,(y_i - b - w\,x_i) \;=\; 0$', r'$\dfrac{\partial J}{\partial w} \;=\; -2\sum_{i=1}^{n}\,x_i\,(y_i - b - w\,x_i) \;=\; 0$'],
'e935': r'$w \;=\; \dfrac{S_{xy}}{S_{xx}} \;=\; \dfrac{\sum_i (x_i-\bar{x})(y_i-\bar{y})}{\sum_i (x_i-\bar{x})^2}\,, \qquad b \;=\; \bar{y} - w\,\bar{x}$',
'e936': r'$R^2 \;=\; 1-\dfrac{\mathrm{SSE}}{\mathrm{SST}} \;=\; 1-\dfrac{\sum_i (y_i-\hat{y}_i)^2}{\sum_i (y_i-\bar{y})^2}$',
'e941': r'$\hat{y}_i \;=\; b + w_1 x_{i1} + w_2 x_{i2} + \cdots + w_p x_{ip}$',
'e942': rf'$\hat{{\mathbf{{y}}}} \;=\; {X}{w}$',
'e943': rf'$J({w}) \;=\; \Vert {y}-{X}{w} \Vert^2 \;=\; ({y}-{X}{w}){T}({y}-{X}{w})$',
'e944': rf'$\nabla J({w}) \;=\; -2\,{X}{T}({y}-{X}{w})$',
'e945': rf'${X}{T}{X}\,{wh} \;=\; {X}{T}{y}$',
'e946': rf'${wh} \;=\; ({X}{T}{X})^{{-1}}{X}{T}{y}$',
'e947': rf'${X}{T}\mathbf{{r}} \;=\; {X}{T}({y}-{X}{wh}) \;=\; \mathbf{{0}}$',
'e948': rf'${wh} \;=\; (\mathbf{{X}}_c{T}\mathbf{{X}}_c)^{{-1}}\mathbf{{X}}_c{T}\mathbf{{y}}_c\,, \qquad b \;=\; \bar{{y}}-\bar{{\mathbf{{x}}}}{T}{wh}$',
'e949': rf'${X} \;=\; \mathbf{{Q}}\mathbf{{R}}\,, \qquad \mathbf{{R}}\,{wh} \;=\; \mathbf{{Q}}{T}{y}$',
'e9410': rf'${X} \;=\; \mathbf{{U}}\,\Sigma\,\mathbf{{V}}{T}, \qquad {wh} \;=\; \sum_{{\sigma_j>0}} \dfrac{{\mathbf{{u}}_j{T}{y}}}{{\sigma_j}}\;\mathbf{{v}}_j$',
'e9101': rf'$y_i \;=\; b + {w}{T}\mathbf{{x}}_i + \varepsilon_i\,, \qquad \varepsilon_i \sim \mathcal{{N}}(0,\sigma^2)$',
'e9102': rf'$\ln L({w},b) \;=\; -\dfrac{{n}}{{2}}\ln(2\pi\sigma^2) \;-\; \dfrac{{1}}{{2\sigma^2}}\sum_{{i=1}}^{{n}}\,(y_i - b - {w}{T}\mathbf{{x}}_i)^2$',
}
E0=dict(E); E0.pop('e934x'); mkeq(E0,'.')
l1,l2=E.pop('e934x')
fig=plt.figure(figsize=(.01,.01)); fig.text(0,0,l1+'\n'+l2,fontsize=11.5,linespacing=2.6); fig.savefig('e934.png',dpi=300,bbox_inches='tight',pad_inches=0.02,facecolor='white'); plt.close(fig)
M=lambda rows:[[str(v) for v in r] for r in rows]
matimg('m951',[rf'${X}=$',M([[1,1],[1,2],[1,3],[1,4],[1,5]]),rf'$,\quad {y}=$',M([[3],[4],[6],[7],[10]]),rf'$,\quad {w}=$',M([['b'],['w']])])
matimg('m952',[rf'${X}{T}{X}=$',M([['n',r'\sum x_i'],[r'\sum x_i',r'\sum x_i^2']]),'$=$',M([[5,15],[15,55]]),rf'$,\quad {X}{T}{y}=$',M([[r'\sum y_i'],[r'\sum x_i y_i']]),'$=$',M([[30],[107]])])
matimg('m953',[rf'$({X}{T}{X})^{{-1}}=\frac{{1}}{{5\times 55-15\times 15}}$',M([[55,-15],[-15,5]]),r'$=\frac{1}{50}$',M([[55,-15],[-15,5]])])
matimg('m954',[rf'${wh}=\frac{{1}}{{50}}$',M([[55,-15],[-15,5]]),M([[30],[107]]),r'$=\frac{1}{50}$',M([[r'55\times 30-15\times 107'],[r'-15\times 30+5\times 107']]),r'$=\frac{1}{50}$',M([[45],[85]]),'$=$',M([['0.9'],['1.7']])])
print('eqs ok')
