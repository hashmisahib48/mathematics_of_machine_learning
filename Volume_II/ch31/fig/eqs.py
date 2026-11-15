# Equation images of Chapter 31. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq, matimg
T=r'^{\mathrm{T}}'
E={
'e3131': r'$a \;=\; \varphi\!\left(\mathbf{w}^{\mathrm{T}}\mathbf{x} + b\right) \;=\; \varphi\!\left(\sum_{i=1}^{p} w_i\,x_i + b\right)$',
'e3132': rf'$\mathbf{{Z}}_{{l}} \;=\; \mathbf{{A}}_{{l-1}}\,\mathbf{{W}}_{{l}} + \mathbf{{1}}\,\mathbf{{b}}_{{l}}{T}\,, \qquad \mathbf{{A}}_{{l}} \;=\; \varphi_{{l}}(\mathbf{{Z}}_{{l}})\,, \qquad \mathbf{{A}}_{{0}} \;=\; \mathbf{{X}}$',
'e3133': r'$\varphi(z) \;=\; z \,, \qquad \dfrac{1}{1+e^{-z}} \,, \qquad \tanh z \,, \qquad \max(0,\,z)$',
'e3134': r'$\varphi^{\prime}(z) \;=\; 1 \,, \qquad a\,(1-a) \,, \qquad 1-a^{2} \,, \qquad [\,z>0\,]$',
'e3135': r'$\hat{y} = z \,, \qquad p = \dfrac{1}{1+e^{-z}} \,, \qquad p_c = \dfrac{e^{z_c}}{\sum_{k} e^{z_k}} \,, \qquad \hat{y} = e^{z}$',
'e3136': r'$J \;=\; \dfrac{1}{n}\sum_{i=1}^{n}\frac{1}{2}\left(\hat{y}_i-y_i\right)^{2} \;+\; \dfrac{\alpha}{2n}\sum_{l}\left\Vert\mathbf{W}_{l}\right\Vert_{F}^{2}$',
'e3137': r'$J \;=\; -\dfrac{1}{n}\sum_{i=1}^{n}\left[\,y_i\ln p_i + (1-y_i)\ln(1-p_i)\,\right] \;+\; \dfrac{\alpha}{2n}\sum_{l}\left\Vert\mathbf{W}_{l}\right\Vert_{F}^{2}$',
'e3138': r'$J \;=\; -\dfrac{1}{n}\sum_{i=1}^{n}\sum_{c=1}^{K} y_{ic}\,\ln p_{ic} \;+\; \dfrac{\alpha}{2n}\sum_{l}\left\Vert\mathbf{W}_{l}\right\Vert_{F}^{2}$',
'e3141': [r'$\dfrac{\partial\ell}{\partial z} \;=\; \dfrac{\partial\ell}{\partial p}\,\dfrac{\mathrm{d}p}{\mathrm{d}z} \;=\; \left(-\dfrac{y}{p}+\dfrac{1-y}{1-p}\right)p\,(1-p) \;=\; p-y$',
          r'$\boldsymbol{\delta}_{\mathrm{out}} \;=\; \dfrac{\partial\ell}{\partial\mathbf{z}_{\mathrm{out}}} \;=\; \hat{\mathbf{y}}-\mathbf{y}$'],
'e3142': r'$\dfrac{\partial\ell}{\partial W^{(1)}_{ij}} \;=\; \dfrac{\partial\ell}{\partial z_2}\,\dfrac{\partial z_2}{\partial h_j}\,\dfrac{\partial h_j}{\partial z_{1j}}\,\dfrac{\partial z_{1j}}{\partial W^{(1)}_{ij}} \;=\; \delta_2\;W^{(2)}_{j}\;\varphi^{\prime}(z_{1j})\;x_i$',
'e3143': rf'$\boldsymbol{{\delta}}_{{l}} \;=\; \left(\boldsymbol{{\delta}}_{{l+1}}\,\mathbf{{W}}_{{l+1}}{T}\right)\odot\varphi_{{l}}^{{\prime}}(\mathbf{{z}}_{{l}})$',
'e3144': rf'$\dfrac{{\partial\ell}}{{\partial\mathbf{{W}}_{{l}}}} \;=\; \mathbf{{a}}_{{l-1}}{T}\,\boldsymbol{{\delta}}_{{l}}\,, \qquad\qquad \dfrac{{\partial\ell}}{{\partial\mathbf{{b}}_{{l}}}} \;=\; \boldsymbol{{\delta}}_{{l}}$',
'e3145': rf'$\dfrac{{\partial J}}{{\partial\mathbf{{W}}_{{l}}}} \;=\; \dfrac{{1}}{{n}}\left(\mathbf{{A}}_{{l-1}}{T}\,\boldsymbol{{\Delta}}_{{l}} + \alpha\,\mathbf{{W}}_{{l}}\right)\,, \qquad \dfrac{{\partial J}}{{\partial\mathbf{{b}}_{{l}}}} \;=\; \dfrac{{1}}{{n}}\sum_{{i=1}}^{{n}}\boldsymbol{{\delta}}_{{l}}^{{(i)}}$',
'e3151': [r'$W_{ij},\; b_j \;\sim\; \mathrm{Uniform}\left(-B,\,B\right), \qquad B \;=\; \sqrt{\dfrac{f}{n_{\mathrm{in}}+n_{\mathrm{out}}}}$',
          r'$f = 6 \quad (f = 2 \ \mathrm{if\ the\ hidden\ activation\ is\ logistic})$'],
'e3152': [r'$\mathbf{v} \;\leftarrow\; \mu\,\mathbf{v} - \eta\,\mathbf{g}\,, \qquad \mathbf{w} \;\leftarrow\; \mathbf{w} + \mathbf{v} \qquad\qquad \mathrm{(momentum)}$',
          r'$\mathbf{v} \;\leftarrow\; \mu\,\mathbf{v} - \eta\,\mathbf{g}\,, \qquad \mathbf{w} \;\leftarrow\; \mathbf{w} + \mu\,\mathbf{v} - \eta\,\mathbf{g} \qquad\ \ \ \mathrm{(Nesterov)}$'],
'e3153': [r'$\eta_t \;=\; \eta_0 \quad \mathrm{(constant)}\,, \qquad \eta_t \;=\; \dfrac{\eta_0}{(t+1)^{q}} \quad \mathrm{(invscaling)}$',
          r'$\eta \;\leftarrow\; \dfrac{\eta}{5} \quad \mathrm{(adaptive,\ when\ the\ loss\ stops\ improving)}$'],
'e3154': [r'$\mathbf{m} \;\leftarrow\; \beta_1\,\mathbf{m} + (1-\beta_1)\,\mathbf{g}\,, \qquad \mathbf{s} \;\leftarrow\; \beta_2\,\mathbf{s} + (1-\beta_2)\,\mathbf{g}^{2}$',
          r'$\eta_t \;=\; \eta\,\dfrac{\sqrt{1-\beta_2^{\,t}}}{1-\beta_1^{\,t}}\,, \qquad \mathbf{w} \;\leftarrow\; \mathbf{w} - \eta_t\,\dfrac{\mathbf{m}}{\sqrt{\mathbf{s}}+\varepsilon}$'],
'e31111': rf'$E(\mathbf{{v}},\mathbf{{h}}) \;=\; -\mathbf{{b}}{T}\mathbf{{v}} - \mathbf{{c}}{T}\mathbf{{h}} - \mathbf{{h}}{T}\mathbf{{W}}\,\mathbf{{v}}$',
'e31112': r'$P(\mathbf{v},\mathbf{h}) \;=\; \dfrac{e^{-E(\mathbf{v},\mathbf{h})}}{Z}\,, \qquad Z \;=\; \sum_{\mathbf{v},\mathbf{h}} e^{-E(\mathbf{v},\mathbf{h})}$',
'e31113': rf'$F(\mathbf{{v}}) \;=\; -\mathbf{{b}}{T}\mathbf{{v}} - \sum_{{j=1}}^{{m}}\ln\left(1+e^{{\,c_j+\mathbf{{W}}_{{j}}\mathbf{{v}}}}\right), \qquad P(\mathbf{{v}}) \;=\; \dfrac{{e^{{-F(\mathbf{{v}})}}}}{{Z}}$',
'e31114': r'$P(h_j=1\,|\,\mathbf{v}) \;=\; \sigma\!\left(c_j+\mathbf{W}_{j}\mathbf{v}\right)\,, \qquad P(v_i=1\,|\,\mathbf{h}) \;=\; \sigma\!\left(b_i+\sum_{j}W_{ji}\,h_j\right)$',
'e31115': r'$\dfrac{\partial\ln P(\mathbf{v})}{\partial W_{ji}} \;=\; \mathbb{E}_{\mathrm{data}}\left[h_j\,v_i\right] - \mathbb{E}_{\mathrm{model}}\left[h_j\,v_i\right]$',
'e31116': rf'$\mathbf{{W}} \;\leftarrow\; \mathbf{{W}} + \dfrac{{\eta}}{{m_b}}\left(\mathbf{{H}}_{{\mathrm{{pos}}}}{T}\mathbf{{V}}_{{\mathrm{{pos}}}} - \mathbf{{H}}_{{\mathrm{{neg}}}}{T}\mathbf{{V}}_{{\mathrm{{neg}}}}\right)$',
'e31117': r'$\mathbf{b} \;\leftarrow\; \mathbf{b} + \dfrac{\eta}{m_b}\sum\left(\mathbf{v}_{\mathrm{pos}}-\mathbf{v}_{\mathrm{neg}}\right)\,, \qquad \mathbf{c} \;\leftarrow\; \mathbf{c} + \dfrac{\eta}{m_b}\sum\left(\mathbf{h}_{\mathrm{pos}}-\mathbf{h}_{\mathrm{neg}}\right)$',
'e31118': r'$\mathrm{PL}(\mathbf{v}) \;=\; d\,\ln\sigma\!\left(F(\tilde{\mathbf{v}})-F(\mathbf{v})\right)$',
'e31121': r'$g(x) \;=\; \max(0,\,2x) - \max(0,\,4x-2)\,, \qquad x\in[0,1]$',
}
mkeq(E,'.')
matimg('m3161',[r'$\mathbf{x}=$',[['2','-1']],r'$,\ \ \mathbf{W}^{(1)}=$',[['0.5','1.0'],['-1.0','0.5']],r'$,\ \ \mathbf{b}^{(1)}=$',[['0.5','-0.5']]])
matimg('m3163',[r'$\mathbf{W}^{(2)}=$',[['1'],['-1']],r'$,\ \ b^{(2)}=0$'])
matimg('m3162',[r'$\mathbf{z}_1=\mathbf{x}\,\mathbf{W}^{(1)}+\mathbf{b}^{(1)}=$',[['2','-1']],[['0.5','1.0'],['-1.0','0.5']],r'$+$',[['0.5','-0.5']],r'$=$',[['2.5','1.0']]])
print('eqs ok')
