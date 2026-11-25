"""The blackboards in the sidebar of Chapter 3: the small steps of the derivations."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from slate import hang
B = {
 'b1': [('h', 'Counting the features'),
        ('t', 'Degree 2 in $p$ features gives the $p$ features, their $p$ squares and the $p(p-1)/2$ products of pairs:'),
        ('m', r'$p+p+\dfrac{p(p-1)}{2}=\dfrac{p(p+3)}{2}$'),
        ('t', 'For the 10 diabetes features:'),
        ('m', r'$10(13)/2=65$')],
 'b2': [('h', 'Derivative of the penalty'),
        ('t', 'Only the term $k=j$ of the sum contains $w_j$, so'),
        ('m', r'$\dfrac{\partial}{\partial w_j}\,\alpha\sum_k w_k^2=2\alpha\,w_j$'),
        ('t', 'and this is the last term of Eq. 4.2-2.')],
 'b3': [('h', 'Shrinkage of one slope'),
        ('t', 'For the five students, $S_{xy}=51$ and $S_{xx}=10$:'),
        ('m', r'$\alpha=0:\ w=5.1$'),
        ('m', r'$\alpha=10:\ w=51/20=2.55$'),
        ('m', r'$\alpha=100:\ w=51/110=0.46$')],
 'b4': [('h', 'The first term of Eq. 4.2-6'),
        ('t', r'With $A=\tilde{X}^{\mathsf{T}}\tilde{X}$, for any $\mathbf{d}$,'),
        ('m', r'$\mathbf{d}^{\mathsf{T}}A\mathbf{d}=\sum_i\left(\sum_k\tilde x_{ik}d_k\right)^2$'),
        ('t', r'the sum of the squares of the $n$ numbers $\sum_k\tilde x_{ik}d_k$.')],
 'b5': [('h', 'Proof of the minimum'),
        ('t', r'Change the weights by $\mathbf{d}$ and let'),
        ('m', r'$\mathbf{r}=\mathbf{c}-(A+\alpha I)\mathbf{w}$'),
        ('m', r'$J(\mathbf{w}+\mathbf{d})=J(\mathbf{w})-2\mathbf{d}^{\mathsf{T}}\mathbf{r}+|\tilde X\mathbf{d}|^2+\alpha|\mathbf{d}|^2$'),
        ('t', r'By Eq. 4.2-4, $\mathbf{r}=\mathbf{0}$; for $\mathbf{d}\neq\mathbf{0}$ the last two terms are positive, so'),
        ('m', r'$J(\mathbf{w}+\mathbf{d})>J(\mathbf{w})$')],
}
if __name__ == '__main__':
    for k, v in B.items():
        hang('H3', v, os.path.join(HERE, 'png', k + '.png')); print(k)
