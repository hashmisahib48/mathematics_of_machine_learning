import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from slate import hang
B = {
 'b1': [('h', 'Gini as a chance of error'),
        ('t', 'Draw one observation of the node, and label it at random with the node’s class fractions. The label is wrong with probability'),
        ('m', r'$\sum_c p_c\,(1-p_c)$'),
        ('t', 'which is the Gini impurity: zero for a pure node. Since the fractions add up to 1, it equals $1-\\sum_c p_c^2$.')],
 'b2': [('h', 'Why only midpoints'),
        ('t', 'Every threshold between two neighboring values $v_k$ and $v_{k+1}$ sends the same points left and right, so only $m-1$ different splits exist. The midpoint is a convention that leaves a margin on both sides.')],
 'b3': [('h', 'Importance by hand'),
        ('t', 'Weighted impurities $n\\,I$:'),
        ('m', r'$\mathrm{root}{:}\;8(0.4688)=3.75$'),
        ('m', r'$x_2{:}\;3.75-[4(0)+4(0.375)]=2.25$'),
        ('m', r'$x_1{:}\;4(0.375)-[1(0)+3(0)]=1.5$'),
        ('t', 'Normalized: $x_2$ 0.6, $x_1$ 0.4.')],
 'b5': [('h', 'Why a split never raises I'),
        ('t', 'The class fractions of the parent are the average of those of the children, with weights $n_L/n$ and $n_R/n$. For a concave $I$, the impurity of an average is at least the average of the impurities, so $I_{\\mathrm{split}}\\leq I$. If $I$ is strictly concave, equality needs both children to keep the parent’s fractions. The error rate is not strictly concave and can stay unchanged.')],
 'b6': [('h', 'Squared error of a split'),
        ('t', 'The product $n\\,\\mathrm{MSE}$ is the sum of squared errors of a node. That of the parent is fixed, so minimizing $n_L\\mathrm{MSE}_L+n_R\\mathrm{MSE}_R$, the total squared error after the split, gives the largest decrease.')],
 'b4': [('h', 'The work of a split'),
        ('t', 'A node with $n$ points and $p$ features has up to $p(n-1)$ candidate splits. Sorting each feature once and moving the points from right to left one at a time evaluates all of them in about $p\\,n\\log n$ operations.')],
}
if __name__ == '__main__':
    for k, v in B.items():
        hang('H3', v, os.path.join(HERE, 'png', k + '.png')); print(k)
