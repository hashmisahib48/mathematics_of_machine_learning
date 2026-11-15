# Equation and matrix images of Chapter 2. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
import numpy as np
from mkeq import mkeq, matimg
import matplotlib.pyplot as plt
T=r'^{\mathrm{T}}'; b=lambda c: rf'\mathbf{{{c}}}'
A=b('A'); B=b('B'); S=b('S'); I=b('I'); a=b('a'); x=b('x'); y=b('y'); u=b('u'); v=b('v'); w=b('w'); c=b('c'); r=b('r'); p=b('p'); q=b('q')
E={
'e211': rf'${w} \;=\; c_1{v}_1 + c_2{v}_2 + \cdots + c_k{v}_k$',
'e222': rf'$\Vert{x}\Vert_{{q}} \;=\; \left(\;\sum_{{j=1}}^{{p}} |x_j|^{{q}}\right)^{{1/q}}$',
'e223': rf'$d({x},{b("z")}) \;=\; \Vert {x}-{b("z")}\Vert_2 \;=\; \sqrt{{(x_1-z_1)^2+\cdots+(x_p-z_p)^2}}$',
'e231': rf'${a}{T}{b("b")} \;=\; a_1 b_1 + a_2 b_2 + \cdots + a_p b_p \;=\; \sum_{{j=1}}^{{p}} a_j b_j$',
'e232': rf'$\Vert {a}-{b("b")}\Vert^2 \;=\; \Vert{a}\Vert^2 - 2\,{a}{T}{b("b")} + \Vert{b("b")}\Vert^2$',
'e233': rf'${a}{T}{b("b")} \;=\; \Vert{a}\Vert\,\Vert{b("b")}\Vert\,\cos\theta$',
'e234': rf'$\cos\theta \;=\; \dfrac{{{a}{T}{b("b")}}}{{\Vert{a}\Vert\,\Vert{b("b")}\Vert}}$',
'e241': rf'${a}{T}\,({b("b")} - c\,{a}) \;=\; 0 \quad\Longrightarrow\quad {a}{T}{b("b")} - c\,{a}{T}{a} \;=\; 0$',
'e242': rf'$c \;=\; \dfrac{{{a}{T}{b("b")}}}{{{a}{T}{a}}}\,, \qquad {p} \;=\; c\,{a} \;=\; \dfrac{{{a}{T}{b("b")}}}{{{a}{T}{a}}}\;{a}$',
'e243': rf'${A}{T}{A}\;{c} \;=\; {A}{T}{b("b")}$',
'e244': rf'${A}{T}{r} \;=\; {A}{T}({b("b")}-{A}{c}) \;=\; {b("0")}$',
'e245': rf'${p} \;=\; {A}{c} \;=\; {A}\,({A}{T}{A})^{{-1}}{A}{T}\,{b("b")} \;=\; {b("P")}\,{b("b")}\,, \qquad {b("P")} \;=\; {A}({A}{T}{A})^{{-1}}{A}{T}$',
'e251': rf'$({A}{x})_i \;=\; a_{{i1}}x_1 + a_{{i2}}x_2 + \cdots + a_{{ip}}x_p \;=\; \sum_{{j=1}}^{{p}} a_{{ij}}\,x_j$',
'e252': rf'$({A}{B})_{{ik}} \;=\; \sum_{{j=1}}^{{p}} a_{{ij}}\,b_{{jk}}$',
'e253': rf'$({A}{B}){T} \;=\; {B}{T}{A}{T}$',
'e261': rf'${A}{c} \;=\; c_1{a}_1 + c_2{a}_2 + \cdots + c_p{a}_p$',
'e262': rf'$c_1{a}_1 + c_2{a}_2 + \cdots + c_p{a}_p \;=\; {b("0")} \quad\text{{only when}}\quad c_1 = c_2 = \cdots = c_p = 0$'.replace(r'\text{only when}',r'\;\mathrm{only\;when}\;'),
'e263': rf'$\mathrm{{rank}}({A}) \;\leq\; \min(n,\,p)$',
'e271': rf'${A}{x} \;=\; {b("b")}$',
'e272': rf'${x} \;=\; {A}^{{-1}}{b("b")}\,, \qquad {A}^{{-1}}{A} \;=\; {A}{A}^{{-1}} \;=\; {I}$',
'e281': rf'${b("Q")}{T}{b("Q")} \;=\; {I}\,, \qquad {b("Q")}^{{-1}} \;=\; {b("Q")}{T}$',
'e282': rf'$\Vert{b("Q")}{x}\Vert^2 \;=\; ({b("Q")}{x}){T}({b("Q")}{x}) \;=\; {x}{T}{b("Q")}{T}{b("Q")}{x} \;=\; {x}{T}{x} \;=\; \Vert{x}\Vert^2$',
'e283': rf'${A} \;=\; {b("Q")}{b("R")}\,, \qquad {b("R")}\,\hat{{{w}}} \;=\; {b("Q")}{T}{y}$',
'e291': rf'${S}{v} \;=\; \lambda\,{v}$',
'e292': rf'$\det({S}-\lambda{I}) \;=\; 0 \quad\Longleftrightarrow\quad \lambda^2 - \mathrm{{trace}}({S})\,\lambda + \det({S}) \;=\; 0$'.replace(r'\mathrm{{trace}}',r'\mathrm{trace}'),
'e293': rf'${S} \;=\; {b("Q")}\,\Lambda\,{b("Q")}{T} \;=\; \lambda_1{q}_1{q}_1{T} + \lambda_2{q}_2{q}_2{T} + \cdots + \lambda_p{q}_p{q}_p{T}$',
'e294': rf'${b("C")} \;=\; \dfrac{{1}}{{n-1}}\,{b("X")}_c{T}{b("X")}_c$',
'e2101': rf'${A} \;=\; {b("U")}\,\Sigma\,{b("V")}{T}$',
'e2102': rf'${A}{v}_j \;=\; \sigma_j\,{u}_j\,, \qquad {A}{T}{A} \;=\; {b("V")}\,\Sigma^2\,{b("V")}{T}\,, \qquad \sigma_j \;=\; \sqrt{{\lambda_j({A}{T}{A})}}$',
'e2103': rf'${A}_k \;=\; \sum_{{j=1}}^{{k}} \sigma_j\,{u}_j\,{v}_j{T}$',
'e2104': rf'${A}^{{+}} \;=\; {b("V")}\,\Sigma^{{+}}\,{b("U")}{T} \;=\; \sum_{{\sigma_j>0}} \dfrac{{1}}{{\sigma_j}}\;{v}_j\,{u}_j{T}$',
'e2105': rf'$\mathrm{{cond}}({A}) \;=\; \dfrac{{\sigma_{{\max}}}}{{\sigma_{{\min}}}} \;=\; \dfrac{{\sigma_1}}{{\sigma_r}}$',
'e2106': rf'$\Vert {A}-{A}_k\Vert_F^2 \;=\; \sigma_{{k+1}}^2 + \sigma_{{k+2}}^2 + \cdots + \sigma_r^2$',
}
mkeq(E,'.')
# multi-line equations
def multi(name,lines,ls=2.4):
    fig=plt.figure(figsize=(.01,.01)); fig.text(0,0,'\n'.join(lines),fontsize=11.5,linespacing=ls); fig.savefig(name+'.png',dpi=300,bbox_inches='tight',pad_inches=0.02,facecolor='white'); plt.close(fig)
multi('e221',[rf'$\Vert{x}\Vert_2 \;=\; \sqrt{{x_1^2 + x_2^2 + \cdots + x_p^2}}$',
              rf'$\Vert{x}\Vert_1 \;=\; |x_1| + |x_2| + \cdots + |x_p|$',
              rf'$\Vert{x}\Vert_\infty \;=\; \max_j\, |x_j|$'],2.3)
multi('e284',[rf'${q}_1 \;=\; \dfrac{{{a}_1}}{{\Vert{a}_1\Vert}}$',rf'${v}_2 \;=\; {a}_2 - ({q}_1{T}{a}_2)\,{q}_1\,, \qquad {q}_2 \;=\; \dfrac{{{v}_2}}{{\Vert{v}_2\Vert}}$'],2.6)
multi('e295',[rf'$\lambda_1{v}_1{T}{v}_2 \;=\; (\mathbf{{S}}{v}_1){T}{v}_2 \;=\; {v}_1{T}\mathbf{{S}}{T}{v}_2 \;=\; {v}_1{T}\mathbf{{S}}{v}_2 \;=\; \lambda_2{v}_1{T}{v}_2$'],2.3)
M=lambda rows:[[str(z) for z in r_] for r_ in rows]
# 2x2 inverse formula
matimg('e273',[r'$\mathbf{A}=$',M([['a','b'],['c','d']]),r'$\;\Longrightarrow\;\;\det\mathbf{A}=ad-bc\,,\quad \mathbf{A}^{-1}=\dfrac{1}{ad-bc}$',M([['d','-b'],['-c','a']])])
# Example 2.5-1
matimg('m251',[r'$\mathbf{MN}=$',M([[1,2],[3,4]]),M([[2,0],[1,3]]),r'$=$',M([[r'1\cdot2+2\cdot1',r'1\cdot0+2\cdot3'],[r'3\cdot2+4\cdot1',r'3\cdot0+4\cdot3']]),r'$=$',M([[4,6],[10,12]])])
matimg('m252',[r'$\mathbf{NM}=$',M([[2,0],[1,3]]),M([[1,2],[3,4]]),r'$=$',M([[r'2\cdot1+0\cdot3',r'2\cdot2+0\cdot4'],[r'1\cdot1+3\cdot3',r'1\cdot2+3\cdot4']]),r'$=$',M([[2,4],[10,14]])])
matimg('m253',[r'$\mathbf{M}\mathbf{x}=$',M([[1,2],[3,4]]),M([[1],[1]]),r'$=$',M([[3],[7]]),r'$,\qquad \mathbf{M}^{\mathrm{T}}=$',M([[1,3],[2,4]])])
# Example 2.7-1
matimg('m271',[r'$\mathbf{K}^{-1}=\dfrac{1}{2}$',M([[2,-2],[-3,4]]),r'$=$',M([[1,-1],['-1.5',2]])])
matimg('m272',[r'$\mathbf{x}=\mathbf{K}^{-1}\mathbf{b}=$',M([[1,-1],['-1.5',2]]),M([[10],[8]]),r'$=$',M([[r'1\cdot10-1\cdot8'],[r'-1.5\cdot10+2\cdot8']]),r'$=$',M([[2],[1]])])
# Example 2.4-1
matimg('m241',[r'$\mathbf{A}=$',M([[2,1],[1,0],[0,1]]),r'$,\quad \mathbf{b}=$',M([[3],[3],[3]]),r'$,\quad \mathbf{A}^{\mathrm{T}}\mathbf{A}=$',M([[5,2],[2,2]]),r'$,\quad \mathbf{A}^{\mathrm{T}}\mathbf{b}=$',M([[9],[6]])])
matimg('m242',[r'$\mathbf{p}=\mathbf{A}\mathbf{c}=$',M([[2,1],[1,0],[0,1]]),M([[1],[2]]),r'$=$',M([[4],[1],[2]]),r'$,\qquad \mathbf{r}=\mathbf{b}-\mathbf{p}=$',M([[-1],[2],[1]])])
# Example 2.9-1
matimg('m291',[r'$\mathbf{S}-6\mathbf{I}=$',M([[-1,2],[2,-4]]),r'$,\qquad \mathbf{S}-1\mathbf{I}=$',M([[4,2],[2,1]])])
# Example 2.10-1 : decimals from numpy with fixed sign convention
Am=np.array([[2.,1],[1,0],[0,1]]); s2=np.sqrt(6)
v1=np.array([2.,1])/np.sqrt(5); v2=np.array([1.,-2])/np.sqrt(5); u1=Am@v1/s2; u2=Am@v2
Um=np.c_[u1,u2]; Vm=np.c_[v1,v2]; assert np.allclose(Um@np.diag([s2,1])@Vm.T,Am)
f3=lambda m:[[f'{z:.3f}' if abs(z)>1e-12 else '0' for z in r_] for r_ in m]
matimg('m2101',[r'$\mathbf{A}=$',f3(Um),r'$\;$',[[f'{s2:.3f}','0'],['0','1']],r'$\;$',f3(Vm.T)])
print('eqs ok')
