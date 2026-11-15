# Equation images of Chapter 36. Run: python3 fig/eqs.py
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
'e3621': r'$\Vert \mathbf{x}-\boldsymbol{\mu}_k\Vert^2 \leq \Vert \mathbf{x}-\boldsymbol{\mu}_j\Vert^2 \quad\Longleftrightarrow\quad 2\,(\boldsymbol{\mu}_j-\boldsymbol{\mu}_k)^{T}\mathbf{x} \;\leq\; \Vert\boldsymbol{\mu}_j\Vert^2-\Vert\boldsymbol{\mu}_k\Vert^2$',
'e3631': r'$J(c,\boldsymbol{\mu}) \;=\; \sum_{i=1}^{n}\,\Vert \mathbf{x}_i-\boldsymbol{\mu}_{c_i}\Vert^2 \;=\; \sum_{k=1}^{K}\ \sum_{i:\,c_i=k}\,\Vert \mathbf{x}_i-\boldsymbol{\mu}_k\Vert^2$',
'e3632': r'$\sum_{i\in C}\Vert \mathbf{x}_i-\mathbf{m}\Vert^2 \;=\; \sum_{i\in C}\Vert \mathbf{x}_i-\bar{\mathbf{x}}_C\Vert^2 \;+\; n_C\,\Vert \bar{\mathbf{x}}_C-\mathbf{m}\Vert^2$',
'e3633': r'$\sum_{i=1}^{n}\Vert \mathbf{x}_i-\bar{\mathbf{x}}\Vert^2 \;=\; J \;+\; \sum_{k=1}^{K} n_k\,\Vert \boldsymbol{\mu}_k-\bar{\mathbf{x}}\Vert^2 \qquad (\mathrm{SST}=\mathrm{SSW}+\mathrm{SSB})$',
'e3634': r'$\sum_{i\in C_k}\Vert \mathbf{x}_i-\boldsymbol{\mu}_k\Vert^2 \;=\; \dfrac{1}{2\,n_k}\sum_{i\in C_k}\ \sum_{j\in C_k}\Vert \mathbf{x}_i-\mathbf{x}_j\Vert^2$',
'e3641': r'$c_i \;=\; \arg\min_{k=1,\dots,K}\ \Vert \mathbf{x}_i-\boldsymbol{\mu}_k\Vert^2$',
'e3642': r'$\boldsymbol{\mu}_k \;=\; \dfrac{1}{n_k}\sum_{i:\,c_i=k}\mathbf{x}_i\,, \qquad n_k=\#\{i:\ c_i=k\}$',
'e3643': r'$J(c^{(t)},\boldsymbol{\mu}^{(t)}) \;\geq\; J(c^{(t+1)},\boldsymbol{\mu}^{(t)}) \;\geq\; J(c^{(t+1)},\boldsymbol{\mu}^{(t+1)})$',
'e3651': r'$P(\mathrm{next\ centre}=\mathbf{x}_i) \;=\; \dfrac{D(\mathbf{x}_i)^2}{\sum_{j=1}^{n} D(\mathbf{x}_j)^2}\,, \qquad D(\mathbf{x})^2=\min_{\mathrm{chosen}\ \boldsymbol{\mu}}\Vert \mathbf{x}-\boldsymbol{\mu}\Vert^2$',
'e3652': r'$\mathbb{E}\,[\,J_{\mathrm{seed}}\,] \;\leq\; 8\,(\ln K+2)\ J_{\mathrm{opt}}$',
'e3661': r'$d(\boldsymbol{\mu}_a,\boldsymbol{\mu}_b)\;\geq\; 2\,d(\mathbf{x},\boldsymbol{\mu}_a) \quad\Longrightarrow\quad d(\mathbf{x},\boldsymbol{\mu}_b)\;\geq\; d(\mathbf{x},\boldsymbol{\mu}_a)$',
'e3662': r'$\boldsymbol{\mu} \;\leftarrow\; \dfrac{N\,\boldsymbol{\mu}+\sum_{i\in B}\mathbf{x}_i}{N+m} \;=\; \boldsymbol{\mu}+\dfrac{m}{N+m}\,(\bar{\mathbf{x}}_B-\boldsymbol{\mu})$',
'e36131': r'$\gamma_{ik} \;=\; \dfrac{\exp\left(-\Vert \mathbf{x}_i-\boldsymbol{\mu}_k\Vert^2/2\sigma^2\right)}{\sum_{j=1}^{K}\exp\left(-\Vert \mathbf{x}_i-\boldsymbol{\mu}_j\Vert^2/2\sigma^2\right)}$',
}
mkeq(E, '.'); print('eqs ok')
