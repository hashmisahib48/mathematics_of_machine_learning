"""The blackboards in the sidebar of Chapter 3: the small steps of the derivations."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from slate import hang
B = {
 'b1': [('h', 'Five students by hand'),
        ('t', 'With $\\bar x=3$ and $\\bar y=62$ the deviations are'),
        ('m', r'$\tilde x_i=-2,\,-1,\,0,\,1,\,2$'),
        ('m', r'$\tilde y_i=-10,\,-5,\,-1,\,6,\,10$'),
        ('m', r'$S_{xx}=10,\ S_{yy}=262,\ S_{xy}=51$'),
        ('m', r'$r=51/\sqrt{10\times262}=0.9964$')],
 'b2': [('h', 'Dot product by hand'),
        ('m', r'$\mathbf{u}\cdot\mathbf{v}=3(4)+4(3)=24$'),
        ('m', r'$|\mathbf{u}|=\sqrt{9+16}=5=|\mathbf{v}|$'),
        ('m', r'$\cos\theta=24/25=0.96$'),
        ('m', r'$\theta=16.3^\circ$')],
 'b3': [('h', 'Derivative of one term'),
        ('t', 'Write $u=y_i-wx_i$, so that $du/dw=-x_i$. By the chain rule'),
        ('m', r'$\dfrac{d}{dw}u^2=2u\,\dfrac{du}{dw}=-2x_i(y_i-wx_i)$'),
        ('t', 'For $x=(1,2,3)$, $y=(2,4,7)$:'),
        ('m', r'$w=\dfrac{2+8+21}{1+4+9}=\dfrac{31}{14}$')],
 'b4': [('h', 'A gradient'),
        ('t', 'For $J=w_1^2+3w_2^2$ each partial derivative treats the other weight as a constant:'),
        ('m', r'$\nabla J=(2w_1,\ 6w_2)$'),
        ('t', 'At $(1.5,\\ 0.8)$ the gradient is $(3,\\ 4.8)$.')],
 'b5': [('h', 'Distance by hand'),
        ('t', 'Flowers a and b of Listing 2.4 differ by'),
        ('m', r'$(-1.9,\ 0.3,\ -3.3,\ -1.2)$'),
        ('m', r'$d^2=3.61+0.09+10.89+1.44=16.03$'),
        ('m', r'$d=4.004,\qquad d_1=6.7$')],
}
if __name__ == '__main__':
    for k, v in B.items():
        hang('H3', v, os.path.join(HERE, 'png', k + '.png')); print(k)
