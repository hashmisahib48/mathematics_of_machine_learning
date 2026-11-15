# Equation images of the Chapter 21 solutions. Run: python3 code/eqs.py  (writes sol/ch21/s21*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq, matimg
E = {
# ---- 21A.1
's21a1_1': [r'$\varphi(\mathbf{x}) \;=\; (1,\ \sqrt{2}\cdot2,\ \sqrt{2}\cdot1,\ 2^2,\ \sqrt{2}\cdot2\cdot1,\ 1^2)$',
            r'$\qquad\;\;=\; (1,\ 2.828,\ 1.414,\ 4,\ 2.828,\ 1)$'],
's21a1_2': [r'$\varphi(\mathbf{z}) \;=\; (1,\ \sqrt{2}\cdot1,\ \sqrt{2}\cdot3,\ 1^2,\ \sqrt{2}\cdot1\cdot3,\ 3^2)$',
            r'$\qquad\;\;=\; (1,\ 1.414,\ 4.243,\ 1,\ 4.243,\ 9)$'],
's21a1_3': r'$\varphi(\mathbf{x})^{T}\varphi(\mathbf{z}) \;=\; 1+4+6+4+12+9 \;=\; 36$',
's21a1_4': r'$(\mathbf{x}^{T}\mathbf{z}+1)^{2} \;=\; (2\cdot1+1\cdot3+1)^{2} \;=\; 6^{2} \;=\; 36$',
# ---- 21A.2
's21a2_1': [r'$2\,a_1+2\,a_2 \;=\; 2$', r'$2\,a_1+5\,a_2 \;=\; 3$'],
's21a2_2': r'$3\,a_2 \;=\; 1 \;\Rightarrow\; a_2=\dfrac{1}{3}\,,\qquad a_1 \;=\; \dfrac{2-2a_2}{2} \;=\; \dfrac{2}{3}$',
's21a2_3': r'$\hat{w} \;=\; \sum_i a_i x_i \;=\; \dfrac{2}{3}\cdot1+\dfrac{1}{3}\cdot2 \;=\; \dfrac{4}{3}$',
's21a2_4': r'$\hat{w} \;=\; \dfrac{\mathbf{x}^{T}\mathbf{y}}{\mathbf{x}^{T}\mathbf{x}+\alpha} \;=\; \dfrac{1\cdot2+2\cdot3}{1^2+2^2+1} \;=\; \dfrac{8}{6} \;=\; \dfrac{4}{3}$',
's21a2_5': [r'$f(3) \;=\; \sum_i a_i\,k(x_i,3) \;=\; \dfrac{2}{3}\cdot(1\cdot3)+\dfrac{1}{3}\cdot(2\cdot3) \;=\; 2+2 \;=\; 4$',
            r'$f(3) \;=\; \hat{w}\cdot3 \;=\; \dfrac{4}{3}\cdot3 \;=\; 4$'],
# ---- 21A.3
's21a3_2': [r'$\mathbf{K}\mathbf{c} = (1-e^{-1}+e^{-2},\ 2e^{-1}-1,\ 1-e^{-1}+e^{-2}) = (0.7675,\ -0.2642,\ 0.7675)$',
            r'$\mathbf{c}^{T}\mathbf{K}\mathbf{c} = 0.7675+0.2642+0.7675 = 3-4e^{-1}+2e^{-2} = 1.7992$'],
's21a3_3': r'$\mathbf{K}\,(1,\,0,\,-1)^{T} \;=\; (1-e^{-2})\,(1,\,0,\,-1)^{T} \quad\Rightarrow\quad \lambda \;=\; 1-e^{-2} \;=\; 0.8647$',
's21a3_4': [r'$\mathbf{K}\,(1,\,t,\,1)^{T} \;=\; \lambda\,(1,\,t,\,1)^{T} \;\Longleftrightarrow\; 1+e^{-2}+e^{-1}t=\lambda\,,\quad 2e^{-1}+t=\lambda\,t$',
            r'$\lambda^{2}-(2+e^{-2})\,\lambda+(1-e^{-2}) \;=\; 0$',
            r'$\lambda^{2}-2.1353\,\lambda+0.8647 \;=\; 0 \;\Rightarrow\; \lambda \;=\; 1.5923\,,\ 0.5430$'],
# ---- questions
's21q1_1': r'$\hat{\mathbf{w}} \;=\; \sum_{i=1}^{n} a_i\,\varphi(\mathbf{x}_i)\,,\qquad f(\mathbf{x}) \;=\; \sum_{i=1}^{n} a_i\,k(\mathbf{x}_i,\mathbf{x})\,,\qquad \mathbf{a} \;=\; (\mathbf{K}+\alpha\mathbf{I})^{-1}\mathbf{y}$',
's21q3_1': r'$J(\mathbf{a}+t\,\mathbf{u}) \;=\; J(\mathbf{a})+\lambda\,(\lambda+\alpha)\,t^{2}\,,\qquad \mathbf{K}\mathbf{u}=\lambda\mathbf{u}\,,\ \ \Vert\mathbf{u}\Vert=1$',
}
mkeq(E, '.')
matimg('s21a2_0', [r'$\mathbf{K}=$', [['1', '2'], ['2', '4']], r'$,\qquad \mathbf{K}+\mathbf{I}=$', [['2', '2'], ['2', '5']], r'$,\qquad \mathbf{y}=$', [['2'], ['3']]], '.')
matimg('s21a3_1', [r'$\mathbf{K}=$', [['1', 'e^{-1}', 'e^{-2}'], ['e^{-1}', '1', 'e^{-1}'], ['e^{-2}', 'e^{-1}', '1']], r'$=$',
                   [['1', '0.3679', '0.1353'], ['0.3679', '1', '0.3679'], ['0.1353', '0.3679', '1']]], '.')
print('solution equations ok:', len(E) + 2)
