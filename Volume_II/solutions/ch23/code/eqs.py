# Equation images of the Chapter 23 solutions. Run: python3 code/eqs.py  (writes sol/ch23/s23*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 23A.1
's23a1_1': [r'$L_{\varepsilon}(-1.5)=1.5-0.5=1.0\,,\quad L_{\varepsilon}(-0.4)=0\,,\quad L_{\varepsilon}(0.2)=0$', r'$L_{\varepsilon}(0.9)=0.9-0.5=0.4\,,\quad L_{\varepsilon}(2.1)=2.1-0.5=1.6$'],
's23a1_2': [r'$\sum L_{\varepsilon} \;=\; 1.0+0+0+0.4+1.6 \;=\; 3.0\,,\qquad \sum|r| \;=\; 1.5+0.4+0.2+0.9+2.1 \;=\; 5.1$', r'$\sum r^2 \;=\; 2.25+0.16+0.04+0.81+4.41 \;=\; 7.67$'],
's23a1_3': r'$J \;=\; \dfrac{1}{2}\Vert\mathbf{w}\Vert^2+C\sum L_{\varepsilon} \;=\; \dfrac{1}{2}\times4+2\times3.0 \;=\; 2+6 \;=\; 8.0$',
# ---- 23A.2
's23a2_1': r'$f(1)=y_1+\varepsilon=1.5\,,\qquad f(3)=y_2-\varepsilon=3.5$',
's23a2_2': r'$w+b=1.5\,,\quad 3w+b=3.5 \;\;\Longrightarrow\;\; w=1\,,\quad b=0.5$',
's23a2_3': r'$\beta_1+\beta_2=0\,,\qquad w=\beta_1\times1+\beta_2\times3=2\beta_2=1 \;\;\Longrightarrow\;\; \beta_2=0.5\,,\quad \beta_1=-0.5$',
's23a2_4': r'$f(2) \;=\; 1\times2+0.5 \;=\; 2.5$',
's23a2_5': r'$J \;=\; \dfrac{1}{2}w^2+C\sum L_{\varepsilon} \;=\; \dfrac{1}{2}\times1+0 \;=\; 0.5$',
's23a2_6': r'$W \;=\; -\dfrac{1}{2}w^2+\mathbf{y}^{T}\boldsymbol{\beta}-\varepsilon\sum|\beta_i| \;=\; -0.5+(-0.5+2.0)-0.5\times1 \;=\; 0.5$',
# ---- 23A.3
's23a3_1': [r'$\xi_3=0.9-0.4=0.5\,,\qquad \xi_4^{*}=1.2-0.4=0.8$', r'$\sum\xi \;=\; 0.5+0.8 \;=\; 1.3$'],
's23a3_2': r'$J \;=\; \dfrac{1}{2}\times2.25+2\times1.3 \;=\; 1.125+2.6 \;=\; 3.725$',
# ---- 23A.4
's23a4_1': r'$\sum|\beta_i| \;=\; C\,\nu\,n \;=\; 2\times0.3\times50 \;=\; 30\,,\qquad \nu n \;=\; 15$',
's23a4_2': [r'$12\ \mathrm{bound}:\;\; 12\times C \;=\; 24\,,\qquad 30-24 \;=\; 6\,,\qquad 6/C \;=\; 3$', r'$k\ \mathrm{free}:\;\; \sum_{\mathrm{free}}|\beta_i| \;<\; k\,C \;\Longrightarrow\; k\,C>6 \;\Longrightarrow\; k\geq4$'],
# ---- 23C
's23c1_1': r'$f(x_A)=y_A-\varepsilon\,,\quad f(x_E)=y_E-\varepsilon \;\;\Longrightarrow\;\; w \;=\; \dfrac{y_E-y_A}{x_E-x_A} \;=\; \dfrac{5.9-2.0}{4} \;=\; 0.975$',
's23c1_2': r'$b \;=\; y_A-\varepsilon-w\,x_A \;=\; 2.0-0.5-0.975 \;=\; 0.525$',
's23c2_1': r'$f(1)=y_1+\varepsilon=3+\varepsilon\,,\quad f(5)=y_5-\varepsilon=10-\varepsilon \;\;\Longrightarrow\;\; 4w=7-2\varepsilon\,,\quad w=\dfrac{7-2\varepsilon}{4}$',
's23c2_2': r'$J(w) \;=\; \dfrac{1}{2}w^2+C\,(0.2-4w)\,,\qquad 0\leq w\leq0.05\quad(\varepsilon=3.4,\ b=6.5)$',
# ---- questions
's23q1_1': r'$\mathbf{w} \;=\; \sum_{i\in\mathrm{SV}}\beta_i\,\mathbf{x}_i\,,\qquad f(\mathbf{x}) \;=\; \sum_{i\in\mathrm{SV}}\beta_i\,k(\mathbf{x}_i,\mathbf{x})+b\,,\qquad \beta_i=0\;\;\mathrm{if}\;\;|r_i|<\varepsilon$',
's23q4_1': r'$\hat{f}(\mathbf{x};\,1000\,\mathbf{y},\,C,\,\varepsilon) \;=\; 1000\;\hat{f}\left(\mathbf{x};\,\mathbf{y},\,C/1000,\,\varepsilon/1000\right)$',
}
mkeq(E, '.'); print('solution equations ok:', len(E))
