"""The blackboards in the sidebar of Chapter 3: the small steps of the derivations."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from slate import hang
B = {
 'b1': [('h', 'The mean minimizes'),
        ('t', 'For numbers $y_1,\\ldots,y_k$, set the derivative to zero:'),
        ('m', r'$\dfrac{d}{da}\sum_i(y_i-a)^2=-2\sum_i(y_i-a)=0$'),
        ('m', r'$a=\dfrac{1}{k}\sum_i y_i$'),
        ('t', 'There are $k$ terms, so the second derivative is $2k>0$ and this is a minimum.')],
 'b2': [('h', 'Weights by hand'),
        ('m', r'$v_B=1/1.118=0.894$'),
        ('m', r'$v_F=1/1.581=0.632$'),
        ('m', r'$v_D=1/1.803=0.555$'),
        ('m', r'$\sum v\,y=0.894(15)+0.632(20)+0.555(24)$'),
        ('m', r'$=39.38$'),
        ('m', r'$\sum v=2.082$'),
        ('t', 'Sums use the unrounded weights.')],
 'b3': [('h', 'Cost of one prediction'),
        ('t', 'Each of the $n$ distances needs $p$ subtractions, squares and additions, so one prediction needs about'),
        ('m', r'$3np$'),
        ('t', 'operations in all, plus sorting. For the diabetes data, $n=331$ and $p=10$: about 10 000 per patient.')],
}
if __name__ == '__main__':
    for k, v in B.items():
        hang('H3', v, os.path.join(HERE, 'png', k + '.png')); print(k)
