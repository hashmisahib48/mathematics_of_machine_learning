import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from slate import hang
B = {
 'b1': [('h', 'The mean is the best center'),
        ('t', 'For fixed members of cluster $k$, set the gradient to zero:'),
        ('m', r'$\dfrac{\partial}{\partial\boldsymbol{\mu}}\sum_{i\in C_k}\Vert\mathbf{x}_i-\boldsymbol{\mu}\Vert^2$'),
        ('m', r'$=-2\sum_{i\in C_k}(\mathbf{x}_i-\boldsymbol{\mu})=\mathbf{0}$'),
        ('t', 'so $\\boldsymbol{\\mu}$ is the mean of the members. It is a minimum, as the sum is a convex quadratic in $\\boldsymbol{\\mu}$ with Hessian $2n_k\\mathbf{I}$.')],
 'b2': [('h', 'Why J never increases'),
        ('t', 'The assignment step gives each point its nearest center, so no term of $J$ grows. The update step replaces each center by the mean, which minimizes its cluster’s sum. $J$ cannot increase, and there are finitely many partitions, so the iterations stop.')],
 'b3': [('h', 'Straight boundaries'),
        ('t', 'A point is nearer to $\\boldsymbol{\\mu}_1$ than to $\\boldsymbol{\\mu}_2$ when'),
        ('m', r'$\Vert\mathbf{x}\Vert^2-2\mathbf{x}^{\mathsf{T}}\boldsymbol{\mu}_1+\Vert\boldsymbol{\mu}_1\Vert^2$'),
        ('m', r'$<\Vert\mathbf{x}\Vert^2-2\mathbf{x}^{\mathsf{T}}\boldsymbol{\mu}_2+\Vert\boldsymbol{\mu}_2\Vert^2$'),
        ('t', 'The squared norm of $\\mathbf{x}$ cancels: the condition is linear in $\\mathbf{x}$.')],
}
if __name__ == '__main__':
    for k, v in B.items():
        hang('H3', v, os.path.join(HERE, 'png', k + '.png')); print(k)
