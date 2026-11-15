# Equation images of Chapter 32. Run: python3 fig/eqs.py
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# 32.3
'e3231': r'$\min_{f_1,\ldots,f_n}\;\sum_{i=1}^{n} w_i\,(y_i-f_i)^2 \quad \mathrm{subject\ to}\quad f_1\leq f_2\leq\cdots\leq f_n$',
'e3232': [r'$L \;=\; \dfrac{1}{2}\sum_{i=1}^{n} w_i\,(y_i-f_i)^2 \;+\; \sum_{i=1}^{n-1}\lambda_i\,(f_i-f_{i+1})\,, \qquad \lambda_i\geq 0\,, \quad \lambda_i\,(f_i-f_{i+1}) = 0$',
          r'$\dfrac{\partial L}{\partial f_i} \;=\; w_i\,(f_i-y_i)+\lambda_i-\lambda_{i-1} \;=\; 0\,, \qquad \lambda_0=\lambda_n=0$'],
'e3233': [r'$\lambda_j \;=\; \sum_{i=s}^{j} w_i\,(y_i-f) \;\geq\; 0 \quad (s\leq j<e)\,, \qquad \lambda_e = 0$',
          r'$f \;=\; \dfrac{\sum_{i=s}^{e} w_i\,y_i}{\sum_{i=s}^{e} w_i}$'],
'e3234': r'$v \;=\; \dfrac{W_a\,v_a+W_b\,v_b}{W_a+W_b}\,, \qquad W \;=\; W_a+W_b$',
'e3235': r'$\hat f(x) \;=\; y_k+\dfrac{x-x_k}{x_{k+1}-x_k}\,(y_{k+1}-y_k)\,, \qquad x_k\leq x\leq x_{k+1}$',
# 32.4
'e3241': r'$P\,(y=1\;|\;\hat p=q) \;=\; q \qquad \mathrm{for\ every}\ q\ \mathrm{in}\ [0,1]$',
'e3242': [r'$\mathrm{E}\,[(y-q)^2] \;=\; \pi\,(1-q)^2+(1-\pi)\,q^2 \;=\; \pi\,(1-\pi)+(q-\pi)^2$',
          r'$\mathrm{E}\,[\ell] \;=\; -\pi\ln q-(1-\pi)\ln(1-q)\,, \qquad \dfrac{\mathrm{d}\,\mathrm{E}[\ell]}{\mathrm{d}q} \;=\; -\dfrac{\pi}{q}+\dfrac{1-\pi}{1-q} \;=\; 0 \;\Rightarrow\; q=\pi$'],
'e3243': [r'$\mathrm{BS} \;=\; \mathrm{REL}-\mathrm{RES}+\mathrm{UNC}$',
          r'$\mathrm{REL}=\dfrac{1}{n}\sum_{k} n_k\,(q_k-\bar o_k)^2\,, \quad \mathrm{RES}=\dfrac{1}{n}\sum_{k} n_k\,(\bar o_k-\bar o)^2\,, \quad \mathrm{UNC}=\bar o\,(1-\bar o)$'],
'e3244': r'$P\,(y=1\;|\;f) \;=\; \dfrac{1}{1+e^{\,A f+B}}$',
'e3245': [r'$J(A,B) \;=\; -\sum_{i=1}^{n}\left[\,t_i\ln p_i+(1-t_i)\ln(1-p_i)\,\right]\,, \qquad p_i=\dfrac{1}{1+e^{A f_i+B}}$',
          r'$t_i \;=\; \dfrac{n_{+}+1}{n_{+}+2}\ \ (y_i=1)\,, \qquad t_i \;=\; \dfrac{1}{n_{-}+2}\ \ (y_i=0)$'],
'e3246': r'$\dfrac{\partial J}{\partial A} \;=\; \sum_{i=1}^{n} f_i\,(t_i-p_i)\,, \qquad \dfrac{\partial J}{\partial B} \;=\; \sum_{i=1}^{n} (t_i-p_i)$',
'e3247': [r'$P\,(y=k\;|\;\mathbf{z}) \;=\; \dfrac{e^{\beta z_k}}{\sum_{j=1}^{K} e^{\beta z_j}}\,, \qquad \beta=\dfrac{1}{T}>0$',
          r'$K=2\,,\ \mathbf{z}=(-f,\,f):\qquad P\,(y=1\;|\;f) \;=\; \sigma(2\beta f)$'],
'e3248': r'$p(\mathbf{x}) \;=\; \dfrac{1}{m}\sum_{j=1}^{m} g_j\left(f_j(\mathbf{x})\right)$',
# 32.5
'e3251': [r'$g(s) \;=\; \sum_{i} f_i\,\left(\sigma(s f_i)-t_i\right)\,, \qquad h(s) \;=\; \sum_{i} f_i^{2}\,\sigma(s f_i)\left(1-\sigma(s f_i)\right)$',
          r'$s \;\leftarrow\; s-\dfrac{g(s)}{h(s)}\,, \qquad A=-s\,,\quad B=0$'],
}
# list entries are rendered line by line and stacked, so that tall lines never overlap
from PIL import Image
import tempfile
tmp = tempfile.mkdtemp()
single = {k: v for k, v in E.items() if not isinstance(v, list)}
mkeq(single, '.')
for k, lines in ((k, v) for k, v in E.items() if isinstance(v, list)):
    parts = {f'{k}_{j}': t for j, t in enumerate(lines)}
    mkeq(parts, tmp)
    ims = [Image.open(f'{tmp}/{n}.png').convert('RGB') for n in parts]
    gap = 14; W = max(i.size[0] for i in ims); H = sum(i.size[1] for i in ims) + gap * (len(ims) - 1)
    out = Image.new('RGB', (W, H), 'white'); y = 0
    for i in ims: out.paste(i, (0, y)); y += i.size[1] + gap
    out.save(f'{k}.png', dpi=(300, 300))
