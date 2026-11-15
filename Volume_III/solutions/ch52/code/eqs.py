# Equation images of the Chapter 52 solutions. Run: python3 code/eqs.py  (writes sol/ch52/s52*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 52A.1
's52a1_1': [r'$\mathrm{idf}_{\mathrm{milk}}=\mathrm{idf}_{\mathrm{sugar}}=\ln\dfrac{1+4}{1+2}+1=1.5108$', r'$\mathrm{idf}_{\mathrm{tea}}=\ln\dfrac{1+4}{1+3}+1=1.2231$'],
's52a1_2': r'$\mathbf{w}_1=(1\times1.5108,\ 0,\ 2\times1.2231)=(1.5108,\ 0,\ 2.4463)$',
's52a1_3': [r'$\Vert\mathbf{w}_1\Vert=\sqrt{1.5108^2+2.4463^2}=\sqrt{8.2669}=2.8752$', r'$\mathbf{z}_1=\dfrac{\mathbf{w}_1}{2.8752}=(0.5255,\ 0,\ 0.8508)$'],
's52a1_4': [r'$\mathrm{idf}_{\mathrm{milk}}=\ln\dfrac{4}{2}+1=1.6931\,,\qquad \mathrm{idf}_{\mathrm{tea}}=\ln\dfrac{4}{3}+1=1.2877$', r'$t_{\mathrm{milk}}=1\,,\qquad t_{\mathrm{tea}}=1+\ln 2=1.6931$'],
's52a1_5': [r'$\mathbf{w}_1=(1.6931,\ 0,\ 1.6931\times1.2877)=(1.6931,\ 0,\ 2.1802)$', r'$\Vert\mathbf{w}_1\Vert=\sqrt{7.6202}=2.7605$'],
's52a1_6': r'$\mathbf{z}_1=\dfrac{(1.6931,\ 0,\ 2.1802)}{2.7605}=(0.6134,\ 0,\ 0.7898)$',
# ---- 52A.2
's52a2_1': r'$E[\mathrm{occupied}]=m\left[1-\left(1-\dfrac{1}{m}\right)^{V}\right]\,,\qquad V=216$',
's52a2_2': [r'$m=64:\quad \left(\dfrac{63}{64}\right)^{216}=e^{216\,\ln(63/64)}=e^{-3.4016}=0.0333$', r'$E[\mathrm{occupied}]=64\,(1-0.0333)=61.87$'],
's52a2_3': [r'$m=1024:\quad \left(\dfrac{1023}{1024}\right)^{216}=0.8097$', r'$E[\mathrm{occupied}]=1024\,(1-0.8097)=194.82$'],
's52a2_4': [r'$\mathrm{lost}=V-E[\mathrm{occupied}]$', r'$m=64:\ \ 216-61.87=154.13$', r'$m=1024:\ \ 216-194.82=21.18$'],
# ---- 52B.1
's52b1_1': [r'$\mathrm{CSR}=\mathrm{nnz}\times(8+4)+4\,(n+1)\ \mathrm{bytes}$', r'$(\mathrm{data\ int64,\ indices\ int32,\ indptr\ int32})$'],
's52b1_2': [r'$(1,1):\ \ 419\times12+4\times49=5028+196=5224$', r'$\mathrm{dense}=8\times48\times216=82944$'],
's52b1_3': [r'$(1,2):\ \ 838\times12+4\times49=10056+196=10252$', r'$\mathrm{dense}=8\times48\times575=220800$'],
# ---- 52C.1
's52c1_1': [r'$P(\mathrm{next\ token\ unseen})\;\approx\;\dfrac{N_1}{N}$', r'$N_1=\mathrm{words\ seen\ once\ in\ training}\,,\quad N=\mathrm{training\ tokens}$'],
# ---- questions
's52q1_1': [r'$\mathrm{idf}^{\,\mathrm{book}}_{j}=\ln\dfrac{n}{\mathrm{df}_j}\ \Rightarrow\ \ 0\ \ \mathrm{if}\ \mathrm{df}_j=n$', r'$\mathrm{idf}^{\,\mathrm{library}}_{j}=\ln\dfrac{1+n}{1+\mathrm{df}_j}+1\ \Rightarrow\ \ 1\ \ \mathrm{if}\ \mathrm{df}_j=n$'],
's52q2_1': [r'$\mathrm{words\ without\ own\ bucket}\;=\;V-m\left[1-\left(1-\dfrac{1}{m}\right)^{V}\right]$', r'$\approx\;\dfrac{V^{2}}{2m}\qquad(m\gg V)$'],
}
mkeq(E, '.'); print('solution equations ok:', len(E))
