"""The blackboards in the sidebar of Chapter 3: the small steps of the derivations."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from slate import hang
B = {
 'b1': [('h', 'Errors of a line'),
        ('t', 'For the line $\\hat y=5.1x+46.7$ and the five students of Figure 1.1-1:'),
        ('m', r'$e_i=y_i-\hat y_i=0.2,\;0.1,\;-1.0,\;0.9,\;-0.2$'),
        ('m', r'$\mathrm{MSE}=\dfrac{0.04+0.01+1+0.81+0.04}{5}=0.38$'),
        ('m', r'$\mathrm{RMSE}=\sqrt{0.38}=0.62\ \mathrm{marks}$')],
 'b3': [('h', 'R² of a constant'),
        ('t', 'A model that always predicts the mean $\\bar{y}$ of the targets has'),
        ('m', r'$\sum_{i=1}^{n}(y_i-\hat y_i)^2=\sum_{i=1}^{n}(y_i-\bar y)^2$'),
        ('t', 'so the fraction in Eq. 1.3-4 is 1 and'),
        ('m', r'$R^2=1-1=0$')],
}
if __name__ == '__main__':
    for k, v in B.items():
        hang('H3', v, os.path.join(HERE, 'png', k + '.png')); print(k)
