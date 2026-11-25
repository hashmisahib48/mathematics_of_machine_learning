import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from slate import hang
B = {
 'b1': [('h', 'Inverting the logistic'),
        ('t', 'From $p=1/(1+e^{-t})$:'),
        ('m', r'$1+e^{-t}=1/p$'),
        ('m', r'$e^{-t}=(1-p)/p$'),
        ('m', r'$t=\ln\dfrac{p}{1-p}$'),
        ('t', 'The score is the logarithm of the odds.')],
 'b2': [('h', 'Derivative of the logistic'),
        ('m', r"$\sigma'(t)=\dfrac{e^{-t}}{(1+e^{-t})^2}$"),
        ('m', r'$=\dfrac{1}{1+e^{-t}}\cdot\dfrac{e^{-t}}{1+e^{-t}}$'),
        ('m', r'$=\sigma(t)\,\left[1-\sigma(t)\right]$'),
        ('t', 'At $t=0$ it equals 1/4.')],
 'b3': [('h', 'Gradient of one term'),
        ('t', 'For one observation, with $\\partial p/\\partial t=p(1-p)$ and $\\partial t/\\partial w_j=x_j$:'),
        ('m', r'$\dfrac{\partial\ell}{\partial t}=-\dfrac{y}{p}p(1-p)+\dfrac{1-y}{1-p}p(1-p)$'),
        ('m', r'$=-y(1-p)+(1-y)p=p-y$'),
        ('m', r'$\dfrac{\partial\ell}{\partial w_j}=(p-y)\,x_j$')],
 'b4': [('h', 'Why the cost is convex'),
        ('t', 'Differentiate Eq. 7.2-6 once more, with $\\tilde{\\mathbf{x}}_i=(1,\\mathbf{x}_i)$:'),
        ('m', r'$H=\dfrac{1}{n}\sum_i p_i(1-p_i)\,\tilde{\mathbf{x}}_i\tilde{\mathbf{x}}_i^{\mathsf{T}}$'),
        ('m', r'$\mathbf{v}^{\mathsf{T}}H\mathbf{v}=\dfrac{1}{n}\sum_i p_i(1-p_i)\,(\tilde{\mathbf{x}}_i^{\mathsf{T}}\mathbf{v})^2$'),
        ('t', 'Each term is at least 0, so $H$ is positive semidefinite and $J$ is convex.')],
}
if __name__ == '__main__':
    for k, v in B.items():
        hang('H3', v, os.path.join(HERE, 'png', k + '.png')); print(k)
