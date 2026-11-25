"""The blackboards in the sidebar of Chapter 3: the small steps of the derivations."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from slate import hang
B = {
 'b1': [('h', 'Gradient by the chain rule'),
        ('t', 'With'),
        ('m', r'$e_i=\tilde y_i-\sum_k\tilde x_{ik}w_k$'),
        ('t', 'only the term $k=j$ contains $w_j$:'),
        ('m', r'$\dfrac{\partial e_i}{\partial w_j}=-\tilde x_{ij},\qquad \dfrac{\partial e_i^2}{\partial w_j}=-2\tilde x_{ij}e_i$'),
        ('t', 'Averaging over the $n$ observations gives Eq. 5.2-1.')],
 'b2': [('h', 'Why the step goes downhill'),
        ('t', 'For a small change $\\mathbf{d}$, the slope in each direction gives'),
        ('m', r'$J(\mathbf{w}+\mathbf{d})\approx J(\mathbf{w})+\sum_j g_j d_j$'),
        ('t', 'With $\\mathbf{d}=-\\eta\\mathbf{g}$ the sum is $-\\eta|\\mathbf{g}|^2$, which is negative unless $\\mathbf{g}=\\mathbf{0}$.')],
 'b3': [('h', 'The one-feature recurrence'),
        ('t', 'The derivative of Eq. 5.2-4 is'),
        ('m', r'$\dfrac{dJ}{dw}=\dfrac{2S_{xx}}{n}\left(w-w^\ast\right)$'),
        ('t', 'so one step gives'),
        ('m', r'$w_{k+1}=w_k-\dfrac{2\eta S_{xx}}{n}\left(w_k-w^\ast\right)$'),
        ('t', 'Subtract $w^\\ast$ from both sides.')],
 'b4': [('h', 'Rates for the students'),
        ('t', 'With $n=5$ and $S_{xx}=10$:'),
        ('m', r'$\rho=1-4\eta$'),
        ('m', r'$\eta=0.1:\ \rho=0.6\qquad \eta=0.25:\ \rho=0$'),
        ('m', r'$\eta=0.45:\ \rho=-0.8\qquad \eta=0.55:\ \rho=-1.2$'),
        ('t', 'Convergence requires $\\eta<0.5$.')],
 'b5': [('h', 'One observation'),
        ('t', 'The squared error of observation $i$ alone has the gradient'),
        ('m', r'$\dfrac{\partial e_i^2}{\partial w_j}=-2\tilde x_{ij}e_i$'),
        ('t', 'Its average over $i$ is the full gradient, Eq. 5.2-1.')],
}
if __name__ == '__main__':
    for k, v in B.items():
        hang('H3', v, os.path.join(HERE, 'png', k + '.png')); print(k)
