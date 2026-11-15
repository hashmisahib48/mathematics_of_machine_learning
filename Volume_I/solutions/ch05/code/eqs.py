# Equation images of the Chapter 5 solutions. Run: python3 code/eqs.py  (writes sol/ch05/s5*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 5A.1
's5a1_1': [r'$f^{\prime}(w) \;=\; 6\,(w-2)$', r'$w_{t+1} \;=\; w_t - 0.1\cdot 6\,(w_t-2) \;=\; 0.4\,w_t + 1.2$'],
's5a1_2': [r'$w_1 \;=\; 0 - 0.6\,(0-2) \;=\; 1.2$',
           r'$w_2 \;=\; 1.2 - 0.6\,(1.2-2) \;=\; 1.2 + 0.48 \;=\; 1.68$',
           r'$w_3 \;=\; 1.68 - 0.6\,(1.68-2) \;=\; 1.68 + 0.192 \;=\; 1.872$'],
's5a1_3': r'$w_{t+1}-2 \;=\; (1-\eta h)\,(w_t-2) \;=\; 0.4\,(w_t-2)$',
's5a1_4': r'$\left|\,1-6\eta\,\right| < 1 \quad\Longleftrightarrow\quad 0 < \eta < \dfrac{2}{6} = \dfrac{1}{3}$',
# ---- 5A.2
's5a2_1': r'$\mathbf{w}_1 \;=\; (0,0) - 0.005\,(-60,\,-214) \;=\; (0.3,\; 1.07)$',
's5a2_2': r'$J(\mathbf{w}_1) \;=\; 2.6569+2.4336+6.2001+5.8564+18.9225 \;=\; 36.0695$'.replace(r'\;',r'\,'),
's5a2_3': r'$\mathbf{g}_1 \;=\; -2\,(\sum r_i\,,\ \sum x_i r_i) \;=\; -2\,(12.45,\ 43.65) \;=\; (-24.9,\ -87.3)$'.replace(r'\;',r'\,'),
's5a2_4': r'$\mathbf{w}_2 \;=\; (0.3,\,1.07) - 0.005\,(-24.9,\,-87.3) \;=\; (0.4245,\; 1.5065)$',
's5a2_5': r'$J(\mathbf{w}_2) \,=\, 1.1428+0.3164+1.1151+0.3020+4.1738 \,=\, 7.0501$',
# ---- 5A.3
's5a3_1': r'$f^{\prime}(w)=e^{w}-2\,,\quad f^{\prime\prime}(w)=e^{w}\,,\qquad w_{t+1} \;=\; w_t-\dfrac{e^{w_t}-2}{e^{w_t}}$',
's5a3_2': r'$w_1 \;=\; 0-\dfrac{1-2}{1} \;=\; 1$',
's5a3_3': r'$w_2 \;=\; 1-\dfrac{2.718282-2}{2.718282} \;=\; 1-0.264241 \;=\; 0.735759$',
's5a3_4': r'$w_3 \,=\, 0.735759-\dfrac{2.087065-2}{2.087065} \,=\, 0.735759-0.041717 \,=\, 0.694042$',
's5a3_5': r'$\dfrac{|w_{t+1}-w^{*}|}{|w_t-w^{*}|^{2}} \;\to\; \dfrac{f^{\prime\prime\prime}(w^{*})}{2f^{\prime\prime}(w^{*})} \;=\; \dfrac{2}{2\times 2} \;=\; 0.5$',
# ---- 5A.4
's5a4_1': r'$L(x,y,\lambda) \;=\; x+2y+\lambda\,(x^{2}+y^{2}-5)$',
's5a4_2': [r'$\partial_x L = 1+2\lambda x = 0\,,\qquad \partial_y L = 2+2\lambda y = 0$', r'$\Longrightarrow\quad x=-\dfrac{1}{2\lambda}\,,\ \ y=-\dfrac{1}{\lambda}$'],
's5a4_3': r'$x^{2}+y^{2} \;=\; \dfrac{1}{4\lambda^{2}}+\dfrac{1}{\lambda^{2}} \;=\; \dfrac{5}{4\lambda^{2}} \;=\; 5 \quad\Longrightarrow\quad \lambda^{2}=\dfrac{1}{4}\,,\ \ \lambda=\pm\dfrac{1}{2}$',
's5a4_4': [r'$\lambda=+\frac{1}{2}:\ \ (x,y)=(-1,-2)\,,\ \ f=-1-4=-5$', r'$\lambda=-\frac{1}{2}:\ \ (x,y)=(1,2)\,,\ \ f=1+4=5$'],
# ---- 5B.1
's5b1_1': [r'$z^{2}-(1+\mu-\eta h)\,z+\mu=0$', r'$(1+\mu-\eta h)^{2}<4\mu \;\Longrightarrow\; |z|=\sqrt{\mu}$'],
# ---- 5C.1
's5c1_1': r'$J(\mathbf{w}^{*}+\mathbf{d}) \;=\; J(\mathbf{w}^{*})+\frac{1}{2}\,\mathbf{d}^{T}\mathbf{H}\,\mathbf{d}+\dots$',
# ---- 5C.2
's5c2_1': [r'$\mathbf{w}(t)=\dfrac{t}{5}\,\mathbf{c}\,,\qquad \lambda(t)=\dfrac{5}{t}-1\qquad (t\leq 5)$', r'$\mathbf{w}=\mathbf{c}\,,\qquad \lambda=0\qquad (t\geq 5)$'],
# ---- questions
's5q1_1': r'$|1-\eta\lambda_k|<1\ \ \mathrm{for\ every}\ k \quad\Longleftrightarrow\quad 0<\eta<\dfrac{2}{\lambda_{\max}}$',
's5q5_1': r'$\lambda \;=\; \dfrac{\Vert\mathbf{c}\Vert}{t}-1\,,\qquad t \;=\; \dfrac{\Vert\mathbf{c}\Vert}{1+\lambda}$',
's5q5_2': r'$t(\alpha) \;=\; \Vert\hat{\mathbf{w}}_{\alpha}\Vert \;=\; \Vert(\mathbf{X}^{T}\mathbf{X}+\alpha\mathbf{I})^{-1}\mathbf{X}^{T}\mathbf{y}\Vert$',
}
mkeq(E, '.'); print('solution equations ok:', len(E))
