# Equation and matrix images of Chapter 3. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq, matimg
import matplotlib.pyplot as plt
T=r'^{\mathrm{T}}'; X=r'\mathbf{X}'; y=r'\mathbf{y}'; w=r'\mathbf{w}'; a=r'\mathbf{a}'; A=r'\mathbf{A}'
def align(name,rows,rh=0.32,fs=11.5,x0=0.42):
    """rows: (lhs, rhs) pairs; rhs begins with '=' and the '=' signs line up."""
    n=len(rows); fig=plt.figure(figsize=(7,rh*n))
    for j,(l,r_) in enumerate(rows):
        yy=1-(j+0.5)/n
        if l: fig.text(x0,yy,l,fontsize=fs,ha='right',va='center')
        fig.text(x0+0.005,yy,r_,fontsize=fs,ha='left',va='center')
    fig.savefig(name+'.png',dpi=300,bbox_inches='tight',pad_inches=0.03,facecolor='white'); plt.close(fig)
M=lambda rows:[[str(v) for v in r] for r in rows]
E={
'e311': r'$f^{\,\prime}(x) \;=\; \lim_{h\to 0}\,\dfrac{f(x+h)-f(x)}{h} \;=\; \dfrac{\mathrm{d}f}{\mathrm{d}x}$',
'e312': r'$y=f(u),\;\; u=g(x) \qquad\Longrightarrow\qquad \dfrac{\mathrm{d}y}{\mathrm{d}x} \;=\; \dfrac{\mathrm{d}y}{\mathrm{d}u}\,\cdot\,\dfrac{\mathrm{d}u}{\mathrm{d}x}$',
'e321': r'$\sigma(z) \;=\; \dfrac{1}{1+e^{-z}}$',
'e341': r'$\dfrac{\partial f}{\partial x_j} \;=\; \lim_{h\to 0}\,\dfrac{f(x_1,\ldots,x_j+h,\ldots,x_p)-f(x_1,\ldots,x_j,\ldots,x_p)}{h}$',
'e342': r'$J(b,w) \;=\; \sum_{i=1}^{n}\,(y_i-b-w\,x_i)^2$',
'e352': r'$D_{\mathbf{u}}f \;=\; \nabla f\cdot\mathbf{u} \;=\; \Vert\nabla f\Vert\,\cos\theta\,, \qquad \Vert\mathbf{u}\Vert=1$',
'e354': r'$\mathbf{w}_{\mathrm{new}} \;=\; \mathbf{w}-\eta\,\nabla J(\mathbf{w})$',
'e361': r'$\dfrac{\mathrm{d}f}{\mathrm{d}t} \;=\; \dfrac{\partial f}{\partial x}\,\dfrac{\mathrm{d}x}{\mathrm{d}t} \;+\; \dfrac{\partial f}{\partial y}\,\dfrac{\mathrm{d}y}{\mathrm{d}t}$',
'e362': r'$\dfrac{\partial L}{\partial w} \;=\; \dfrac{\partial L}{\partial p}\,\dfrac{\partial p}{\partial z}\,\dfrac{\partial z}{\partial u}\,\dfrac{\partial u}{\partial w}\,, \qquad \dfrac{\partial L}{\partial b} \;=\; \dfrac{\partial L}{\partial p}\,\dfrac{\partial p}{\partial z}\,\dfrac{\partial z}{\partial b}$',
'e371': rf'$\nabla(\mathbf{{a}}{T}{w}) \;=\; \mathbf{{a}}$',
'e372': rf'${w}{T}{A}{w} \;=\; \sum_{{j=1}}^{{p}}\;\sum_{{l=1}}^{{p}} A_{{jl}}\,w_j\,w_l$',
'e374': rf'$\nabla({w}{T}{A}{w}) \;=\; ({A}+{A}{T})\,{w} \;=\; 2\,{A}{w}\qquad(\mathbf{{A}}\ \mathrm{{symmetric}})$',
'e375': rf'$J({w}) \;=\; \Vert {y}-{X}{w} \Vert^2 \;=\; {y}{T}{y} \;-\; 2\,{w}{T}{X}{T}{y} \;+\; {w}{T}{X}{T}{X}{w}$',
'e376': rf'$\nabla J({w}) \;=\; -2\,{X}{T}{y} \;+\; 2\,{X}{T}{X}{w} \;=\; -2\,{X}{T}({y}-{X}{w})$',
'e383': rf'$\nabla^2({w}{T}{A}{w}) \;=\; {A}+{A}{T} \;=\; 2\,{A}\qquad(\mathbf{{A}}\ \mathrm{{symmetric}})$',
'e384': rf'$\nabla^2 J({w}) \;=\; 2\,{X}{T}{X}$',
'e387': r'$\dfrac{\mathrm{d}^2}{\mathrm{d}s^2}\,f(\mathbf{w}+s\,\mathbf{u})\vert_{s=0} \;=\; \mathbf{u}^{\mathrm{T}}\mathbf{H}\,\mathbf{u}$',
'e391': r'$f(x+h) \;\approx\; f(x) \;+\; f^{\,\prime}(x)\,h$',
'e392': r'$f(x+h) \;\approx\; f(x) \;+\; f^{\,\prime}(x)\,h \;+\; \frac{1}{2}\,f^{\,\prime\prime}(x)\,h^2$',
'e393': r'$f(\mathbf{w}+\mathbf{d}) \;\approx\; f(\mathbf{w}) \;+\; \nabla f(\mathbf{w})^{\mathrm{T}}\mathbf{d} \;+\; \frac{1}{2}\,\mathbf{d}^{\mathrm{T}}\mathbf{H}\,\mathbf{d}$',
'e394': r'$\mathbf{g}+\mathbf{H}\,\mathbf{d}=\mathbf{0} \qquad\Longrightarrow\qquad \mathbf{d} \;=\; -\mathbf{H}^{-1}\mathbf{g}$',
'e3101': r'$f^{\,\prime}(x) \;\approx\; \dfrac{f(x+h)-f(x-h)}{2h}$',
'e3102': r'$\varepsilon \;=\; \dfrac{\Vert \mathbf{g}_{\mathrm{analytic}}-\mathbf{g}_{\mathrm{numeric}}\Vert}{\Vert \mathbf{g}_{\mathrm{analytic}}\Vert+\Vert \mathbf{g}_{\mathrm{numeric}}\Vert}$',
}
mkeq(E,'.')
dl=r'\dfrac{\mathrm{d}}{\mathrm{d}z}'
align('e322',[(r'$\sigma^{\prime}(z)$',r'$=\;-\,(1+e^{-z})^{-2}\cdot(-e^{-z}) \;=\; \dfrac{e^{-z}}{(1+e^{-z})^2}$'),
              ('',r'$=\;\dfrac{1}{1+e^{-z}}\cdot\dfrac{e^{-z}}{1+e^{-z}}$'),
              ('',r'$=\;\sigma(z)\,(1-\sigma(z))$')],rh=0.5)
align('e323',[(r'$\mathrm{softplus}(z)$',r'$=\;\ln\,(1+e^{z})$'),
              (r'$\mathrm{softplus}^{\prime}(z)$',r'$=\;\dfrac{e^{z}}{1+e^{z}} \;=\; \sigma(z)$')],rh=0.5,x0=0.32)
align('e331',[(r'$J(w)$',r'$=\;\sum_{i=1}^{5}\,(y_i-0.9-w\,x_i)^2 \;=\; 55\,w^2-187\,w+160.05$')],rh=0.5,x0=0.12)
align('e332',[(r'$J^{\prime}(w)$',r'$=\;110\,w-187 \;=\; 0 \qquad\Longrightarrow\qquad w=\dfrac{187}{110}=1.7$'),
              (r'$J^{\prime\prime}(w)$',r'$=\;110 \;>\; 0$')],rh=0.5,x0=0.18)
align('e343',[(r'$\dfrac{\partial J}{\partial b}$',r'$=\;-2\sum_{i=1}^{n}\,r_i$'),
              (r'$\dfrac{\partial J}{\partial w}$',r'$=\;-2\sum_{i=1}^{n}\,x_i\,r_i$')],rh=0.55,x0=0.12)
align('e371b',[(r'$\dfrac{\partial}{\partial w_k}\,(\mathbf{a}^{\mathrm{T}}\mathbf{w})$',r'$=\;\dfrac{\partial}{\partial w_k}\;\sum_{j=1}^{p} a_j\,w_j \;=\; a_k$')],rh=0.55,x0=0.32)
align('e363',[(r'$\dfrac{\partial L}{\partial z}$',r'$=\;\left(-\dfrac{y}{p}+\dfrac{1-y}{1-p}\right)p\,(1-p) \;=\; -y\,(1-p)+(1-y)\,p \;=\; p-y$')],rh=0.6,x0=0.12)
align('e373',[(r'$\dfrac{\partial}{\partial w_k}\,(\mathbf{w}^{\mathrm{T}}\mathbf{A}\mathbf{w})$',r'$=\;\sum_{l=1}^{p} A_{kl}\,w_l \;+\; \sum_{j=1}^{p} A_{jk}\,w_j \;=\; (\mathbf{A}\mathbf{w})_k+(\mathbf{A}^{\mathrm{T}}\mathbf{w})_k$')],rh=0.6,x0=0.32)
align('e386',[(r'$\mathbf{r}(\mathbf{w})$',r'$=\;\mathbf{y}-\mathbf{X}\mathbf{w}\,, \qquad \mathrm{D}\mathbf{r}=-\mathbf{X}$'),
              (r'$\nabla\Vert\mathbf{r}\Vert^2$',r'$=\;2\,(\mathrm{D}\mathbf{r})^{\mathrm{T}}\mathbf{r} \;=\; -2\,\mathbf{X}^{\mathrm{T}}(\mathbf{y}-\mathbf{X}\mathbf{w})$'),
              (r'$\nabla^2\Vert\mathbf{r}\Vert^2$',r'$=\;2\,(\mathrm{D}\mathbf{r})^{\mathrm{T}}(\mathrm{D}\mathbf{r}) \;=\; 2\,\mathbf{X}^{\mathrm{T}}\mathbf{X}$')],rh=0.36,x0=0.2)
matimg('e351',[r'$\nabla f(\mathbf{x}) \;=\;$',M([[r'\partial f/\partial x_1'],[r'\partial f/\partial x_2'],[r'\vdots'],[r'\partial f/\partial x_p']])],rh=0.26)
matimg('e353',[r'$\nabla J(b,w) \;=\;$',M([[r'-2\sum_{i=1}^{n} r_i'],[r'-2\sum_{i=1}^{n} x_i\,r_i']]),r'$\;=\;-2\,\mathbf{X}^{\mathrm{T}}\mathbf{r}$'],rh=0.42)
matimg('e381',[r'$\mathrm{D}\mathbf{f} \;=\;$',M([[r'\partial f_1/\partial w_1',r'\cdots',r'\partial f_1/\partial w_p'],[r'\vdots',r'\,',r'\vdots'],[r'\partial f_m/\partial w_1',r'\cdots',r'\partial f_m/\partial w_p']])],rh=0.3)
matimg('e382',[r'$\mathbf{H} \;=\;$',M([[r'\partial^2 f/\partial w_1^2',r'\partial^2 f/\partial w_1\partial w_2'],[r'\partial^2 f/\partial w_2\partial w_1',r'\partial^2 f/\partial w_2^2']]),r'$,\quad H_{jk}=\dfrac{\partial^2 f}{\partial w_j\,\partial w_k}$'],rh=0.3)
matimg('e385',[r'$\nabla^2 J \;=\; 2\,\mathbf{X}^{\mathrm{T}}\mathbf{X} \;=\; 2$',M([[5,15],[15,55]]),'$=$',M([[10,30],[30,110]])],rh=0.24)
matimg('m361',[r'$\nabla J(0,1) \;=\; -2\,\mathbf{X}^{\mathrm{T}}\mathbf{r} \;=\; -2$',M([[1,1,1,1,1],[1,2,3,4,5]]),M([[2],[2],[3],[3],[5]]),'$=$',M([[-30],[-104]])],rh=0.22)
print('eqs ok')
