import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from slate import hang
B = {
 'b1': [('h', 'The denominator'),
        ('t', 'In Eq. 8.2-1 the denominator'),
        ('m', r'$p(\mathbf{x})=\sum_k P(k)\,p(\mathbf{x}\mid k)$'),
        ('t', 'is the same for every class $c$. Dividing all the numerators by the same positive number does not change which is largest, so it can be omitted when only the class is wanted.')],
 'b2': [('h', 'Class A by hand'),
        ('t', 'Points (2, 6), (3, 8), (4, 7):'),
        ('m', r'$\mu_{A1}=(2+3+4)/3=3$'),
        ('m', r'$\sigma_{A1}^2=(1+0+1)/3=2/3$'),
        ('m', r'$\mu_{A2}=(6+8+7)/3=7$'),
        ('m', r'$\sigma_{A2}^2=(1+1+0)/3=2/3$')],
 'b5': [('h', 'Class B by hand'),
        ('t', 'Points (6, 3), (7, 5), (8, 4), (7, 4):'),
        ('m', r'$\mu_{B1}=(6+7+8+7)/4=7$'),
        ('m', r'$\sigma_{B1}^2=(1+0+1+0)/4=1/2$'),
        ('m', r'$\mu_{B2}=(3+5+4+4)/4=4$'),
        ('m', r'$\sigma_{B2}^2=(1+1+0+0)/4=1/2$')],
 'b3': [('h', 'Why add alpha'),
        ('t', 'The word lunch never occurs in spam. Without smoothing'),
        ('m', r'$\hat\theta_{\mathrm{spam,lunch}}=0/7=0$'),
        ('t', 'and any message containing lunch gets the spam probability 0, whatever its other words. With $\\alpha=1$:'),
        ('m', r'$\hat\theta_{\mathrm{spam,lunch}}=1/15$')],
 'b4': [('h', 'A linear classifier'),
        ('t', 'For two classes, from Eq. 8.2-8:'),
        ('m', r'$L_1-L_0=\ln\dfrac{P(1)}{P(0)}+\sum_j x_j\ln\dfrac{\theta_{1j}}{\theta_{0j}}$'),
        ('t', 'a constant plus weights times counts: the boundary $L_1=L_0$ is a hyperplane, as in Chapter 7.')],
}
if __name__ == '__main__':
    for k, v in B.items():
        hang('H3', v, os.path.join(HERE, 'png', k + '.png')); print(k)
