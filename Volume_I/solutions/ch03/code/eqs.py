# Equation images of the Chapter 3 solutions. Run: python3 code/eqs.py  (writes sol/ch03/s3*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq, matimg
E = {
# ---- 3A.1
's3a1_1': r"$u=3x^{2}+1\,,\qquad \dfrac{\mathrm{d}u}{\mathrm{d}x}=6x\,,\qquad \dfrac{\mathrm{d}f}{\mathrm{d}u}=4u^{3}$",
's3a1_2': r"$f'(x)=4u^{3}\cdot 6x=24x\,(3x^{2}+1)^{3}\,,\qquad f'(1)=24\cdot 4^{3}=1536$",
's3a1_3': r"$g'(x)=(x^{2})'e^{-x}+x^{2}\,(e^{-x})'=2x\,e^{-x}-x^{2}e^{-x}=(2x-x^{2})\,e^{-x}$",
's3a1_4': r"$g'(1)=(2-1)\,e^{-1}=0.3679$",
# ---- 3A.2
's3a2_1': r"$\sigma(0)=\dfrac{1}{1+e^{0}}=\dfrac{1}{2}\,,\qquad \sigma'(0)=\dfrac{1}{2}\cdot\dfrac{1}{2}=0.25$",
's3a2_2': r"$\sigma(2)=\dfrac{1}{1+e^{-2}}=\dfrac{1}{1.1353}=0.8808\,,\qquad 1-\sigma(2)=0.1192$",
's3a2_3': r"$\sigma'(2)=\sigma(2)\,[1-\sigma(2)]=0.8808\times 0.1192=0.1050$",
's3a2_4': r"$\dfrac{\sigma'(0)}{\sigma'(2)}=\dfrac{0.25}{0.1050}=2.38\approx 2.4$",
# ---- 3A.3
's3a3_1': r"$\dfrac{\partial f}{\partial u}=2u+v\,,\qquad \dfrac{\partial f}{\partial v}=u+4v$",
's3a3_7': r"$\nabla f(1,-1)=(2\cdot 1-1\,,\ \ 1+4\cdot(-1))=(1,\,-3)$",
's3a3_2': r"$\dfrac{\partial^{2}f}{\partial u^{2}}=2\,,\qquad \dfrac{\partial^{2}f}{\partial u\,\partial v}=\dfrac{\partial^{2}f}{\partial v\,\partial u}=1\,,\qquad \dfrac{\partial^{2}f}{\partial v^{2}}=4$",
's3a3_4': r"$2u+v=0\,,\quad u+4v=0\ \ \Longrightarrow\ \ v=-2u\,,\ \ u-8u=0\ \ \Longrightarrow\ \ u=v=0$",
's3a3_5': r"$(2-\lambda)(4-\lambda)-1=\lambda^{2}-6\lambda+7=0\,,\qquad \lambda=3\pm\sqrt{2}$",
's3a3_6': r"$\lambda_{1}=3-1.4142=1.586\,,\qquad \lambda_{2}=3+1.4142=4.414$",
# ---- 3A.4
's3a4_1': r"$\nabla J=\left(-2\sum_{i}r_{i}\,,\ \ -2\sum_{i}x_{i}\,r_{i}\right)$",
's3a4_2': r"$\nabla J=\left(-2\times(-5)\,,\ \ -2\times(-18)\right)=(10,\ 36)$",
's3a4_3': r"$b_{\mathrm{new}}=1-0.01\times 10=0.9\,,\qquad w_{\mathrm{new}}=2-0.01\times 36=1.64$",
's3a4_4': r"$J=0.2116+0.0324+0.0324+0.2116+0.8100=1.298$",
's3a4_5': r"$J(\mathbf{v}-\eta\mathbf{g})\ \approx\ J(\mathbf{v})-\eta\,\Vert\mathbf{g}\Vert^{2}+\frac{1}{2}\eta^{2}\,\mathbf{g}^{T}\mathbf{H}\mathbf{g}$",
's3a4_6': r"$7-0.01\times 1396+\frac{1}{2}\,0.01^{2}\times 165160=7-13.960+8.258$",
# ---- 3B
's3b1_1': r"$\nabla\left[J+\alpha\,\Vert\mathbf{v}\Vert^{2}\right]=-2\mathbf{X}^{T}(\mathbf{y}-\mathbf{X}\mathbf{v})+2\alpha\,\mathbf{v}$",
's3b2_1': r"$f(1.5)\ \approx\ f(1)+f'(1)\,h+\frac{1}{2}f''(1)\,h^{2}\,,\qquad h=0.5$",
's3b2_2': r"$1.3133+0.7311\times 0.5+\frac{1}{2}\times 0.1966\times 0.25=1.7034$",
's3b3_1': r"$E(h)\ \approx\ \dfrac{h^{2}}{6}\,\vert\cos 1\vert\ +\ \dfrac{\varepsilon\,\sin 1}{h}$",
# ---- 3C
's3c1_1': r"$f(z)=\frac{1}{2}(z-0.3)^{2}+0.5\,\vert z\vert\,,\qquad f'(0)\ \ni\ (0-0.3)+0.5\,s\,,\quad s\in[-1,\,1]$",
's3c1_2': r"$-0.3+0.5\,s=0\ \ \Longrightarrow\ \ s=0.6\in[-1,\,1]$",
's3c2_1': r"$\dfrac{\partial L}{\partial z}=\dfrac{\partial L}{\partial p}\,\sigma'(z)$",
's3c2_2': r"$L=(p-y)^{2}:\quad \dfrac{\partial L}{\partial z}=2(p-y)\,p(1-p)$",
's3c2_3': r"$L=-y\ln p-(1-y)\ln(1-p):\quad \dfrac{\partial L}{\partial z}=\dfrac{p-y}{p(1-p)}\,p(1-p)=p-y$",
's3c3_1': r"$\cos\theta=\dfrac{(-\nabla J)\cdot\mathbf{d}}{\Vert\nabla J\Vert\,\Vert\mathbf{d}\Vert}=\dfrac{14\times 0.7}{14\times 2.2136}=0.316\,,\qquad \theta=71.6^{\circ}$",
's3c3_2': r"$\mathbf{e}_{t+1}=(\mathbf{I}-\eta\mathbf{H})\,\mathbf{e}_{t}\,,\qquad \mathrm{factor\ per\ step}\ \ 1-\eta\lambda$",
's3c3_3': r"$1-0.01\times 1.69=0.9831\,,\qquad 1-0.01\times 118.31=-0.1831$",
# ---- questions
's3q3_1': r"$f(u,v)=u^{2}-v^{2}\,,\qquad \nabla f(0,0)=\mathbf{0}$",
}
mkeq(E, '.')
matimg('s3a3_3', [r'$\mathbf{H}=$', [['2', '1'], ['1', '4']]], '.')
matimg('s3q3_2', [r'$\mathbf{H}=$', [['2', '0'], ['0', '-2']]], '.')
print('solution equations ok:', len(E) + 2)
