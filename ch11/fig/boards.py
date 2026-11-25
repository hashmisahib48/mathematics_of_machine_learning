import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from slate import hang
B = {
 'b1': [('h', 'Mean 0, deviation 1'),
        ('t', 'Average Eq. 11.2-1 over the training cases:'),
        ('m', r'$\dfrac{1}{n}\sum_i z_{ij}=\dfrac{1}{s_j}\left(\dfrac{1}{n}\sum_i x_{ij}-\bar x_j\right)=0$'),
        ('m', r'$\dfrac{1}{n}\sum_i z_{ij}^2=\dfrac{1}{s_j^2}\cdot\dfrac{1}{n}\sum_i\left(x_{ij}-\bar x_j\right)^2=1$'),
        ('t', 'On new data the scaled feature has a mean and deviation only close to 0 and 1.')],
 'b2': [('h', 'Why harmonic'),
        ('t', 'The reciprocal of $F_1$ is the mean of the reciprocals of $P$ and $R$:'),
        ('m', r'$\dfrac{1}{F_1}=\dfrac{1}{2}\left(\dfrac{1}{P}+\dfrac{1}{R}\right)=\dfrac{P+R}{2PR}$'),
        ('t', 'and so $F_1=2PR/(P+R)$. For $P=1$ and $R=0.1$ the mean is 0.55 but $F_1=0.18$: the small value dominates.')],
 'b3': [('h', 'Area equals pair count'),
        ('t', 'Pass the cases from the highest score down. A positive case moves the curve up by $1/n_{+}$, a negative one right by $1/n_{-}$. A positive case adds a strip of height $1/n_{+}$ and width (negatives ranked below it)/$n_{-}$. Summing the strips gives Eq. 11.2-7. A tie moves the curve diagonally, which counts one half.')],
}
if __name__ == '__main__':
    for k, v in B.items():
        hang('H3', v, os.path.join(HERE, 'png', k + '.png')); print(k)
