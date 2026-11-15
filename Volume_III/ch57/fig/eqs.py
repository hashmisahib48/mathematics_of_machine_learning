# Equation images of Chapter 57. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq
E={
'e5731': [r"$\mathrm{recall}=\dfrac{TP}{TP+FN}\,,\qquad \mathrm{specificity}=\dfrac{TN}{TN+FP}\,,\qquad \mathrm{precision}=\dfrac{TP}{TP+FP}$",
          r"$\mathrm{accuracy}=\dfrac{TP+TN}{TP+FN+FP+TN}\,,\qquad \mathrm{FPR}=1-\mathrm{specificity}=\dfrac{FP}{FP+TN}$"],
'e5732': [r"$\dfrac{1}{F_\beta}\;=\;\dfrac{1}{1+\beta^{2}}\cdot\dfrac{1}{P}\;+\;\dfrac{\beta^{2}}{1+\beta^{2}}\cdot\dfrac{1}{R}\,,\qquad F_\beta=\dfrac{(1+\beta^{2})\,P\,R}{\beta^{2}P+R}$",
          r"$F_\beta\;=\;\dfrac{(1+\beta^{2})\,TP}{(1+\beta^{2})\,TP+\beta^{2}\,FN+FP}\,,\qquad F_1=\dfrac{2\,TP}{2\,TP+FN+FP}$"],
'e5733': [r"$\mathrm{BA}=\dfrac{1}{K}\sum_{k=1}^{K}R_k\,,\qquad \mathrm{BA}_{\mathrm{adj}}=\dfrac{\mathrm{BA}-1/K}{1-1/K}\,,\qquad J=\dfrac{TP}{TP+FN+FP}$"],
'e5734': [r"$\mathrm{micro:}\ \ F_{\beta}=\dfrac{(1+\beta^{2})\sum_k TP_k}{\sum_k\left(\beta^{2}\,n_k+m_k\right)}\,,\qquad \mathrm{macro:}\ \ \dfrac{1}{K}\sum_{k}F_{\beta,k}$",
          r"$\mathrm{weighted:}\ \ \dfrac{\sum_k n_k\,F_{\beta,k}}{\sum_k n_k}\,,\qquad n_k=TP_k+FN_k\,,\quad m_k=TP_k+FP_k$"],
'e5735': [r"$\mathrm{MCC}=\dfrac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$",
          r"$\mathrm{MCC}=\dfrac{c\,n-\sum_k n_k m_k}{\sqrt{\left(n^{2}-\sum_k m_k^{2}\right)\left(n^{2}-\sum_k n_k^{2}\right)}}\,,\qquad c=\sum_k TP_k$",
          r"$\kappa=\dfrac{p_o-p_e}{1-p_e}\,,\qquad p_o=\dfrac{c}{n}\,,\qquad p_e=\sum_k\dfrac{n_k}{n}\cdot\dfrac{m_k}{n}$"],
'e5736': [r"$\mathrm{TPR}(t)=P(S_{+}>t)\,,\qquad \mathrm{FPR}(t)=P(S_{-}>t)$",
          r"$\mathrm{AUC}=\int_{-\infty}^{\infty}\mathrm{TPR}(t)\,\left[-d\,\mathrm{FPR}(t)\right]=\int P(S_{+}>t)\,f_{-}(t)\,dt=P(S_{+}>S_{-})$"],
'e5737': [r"$\mathrm{AUC}=\dfrac{1}{n_{+}n_{-}}\sum_{i\,\mathrm{positive}}\ \sum_{j\,\mathrm{negative}}\left[\,\mathbf{1}(s_i>s_j)+\dfrac{1}{2}\,\mathbf{1}(s_i=s_j)\right]$",
          r"$\mathrm{AUC}=\dfrac{\sum_{i\,\mathrm{positive}}\mathrm{rank}_i-n_{+}(n_{+}+1)/2}{n_{+}\,n_{-}}$"],
'e5738': [r"$S_{+}-S_{-}\sim N(d,\,2)\ \ \Longrightarrow\ \ \mathrm{AUC}=P(S_{+}-S_{-}>0)=\Phi\left(d/\sqrt{2}\right)$"],
'e5739': [r"$\mathrm{AP}=\sum_{j}\left(R_j-R_{j-1}\right)P_j\,,\qquad R_j=\dfrac{TP_j}{n_{+}}\,,\quad P_j=\dfrac{TP_j}{TP_j+FP_j}\,,\quad R_0=0$"],
'e57310': [r"$P_j=\dfrac{\mathrm{TPR}\cdot\pi}{\mathrm{TPR}\cdot\pi+\mathrm{FPR}\cdot(1-\pi)}\,,\qquad \dfrac{P_j}{1-P_j}=\dfrac{\mathrm{TPR}}{\mathrm{FPR}}\cdot\dfrac{\pi}{1-\pi}$"],
'e57311': [r"$\mathrm{log\ loss}=-\dfrac{1}{n}\sum_{i=1}^{n}\ln\,p_{i,\,y_i}\,,\qquad \mathrm{Brier}=\dfrac{c_K}{n}\sum_{i=1}^{n}\sum_{k=1}^{K}\left(\mathbf{1}(y_i=k)-p_{ik}\right)^{2}$",
           r"$c_K=\dfrac{1}{2}\ \ \mathrm{for}\ K=2\ \ \left(\mathrm{Brier}=\dfrac{1}{n}\sum_i(y_i-p_i)^{2}\right)\,,\qquad c_K=1\ \ \mathrm{for}\ K\geq 3$"],
'e57312': [r"$D^{2}_{\mathrm{log}}=1-\dfrac{\sum_i\left[-\ln\,p_{i,\,y_i}\right]}{\sum_i\left[-\ln\,\bar{p}_{y_i}\right]}\,,\qquad \bar{p}_k=\dfrac{n_k}{n}$"],
'e5741': [r"$\mathrm{MAE}=\dfrac{1}{n}\sum_i\left|r_i\right|,\qquad \mathrm{MSE}=\dfrac{1}{n}\sum_i r_i^{2},\qquad \mathrm{RMSE}=\sqrt{\mathrm{MSE}},\qquad r_i=y_i-\hat{y}_i$",
          r"$\mathrm{MAPE}=\dfrac{1}{n}\sum_i\dfrac{\left|r_i\right|}{\max(|y_i|,\,\varepsilon)}\,,\qquad \mathrm{MedAE}=\mathrm{median}_i\left|r_i\right|\,,\qquad \mathrm{max\ error}=\max_i\left|r_i\right|$"],
'e5742': [r"$R^{2}=1-\dfrac{\sum_i r_i^{2}}{\sum_i\left(y_i-\bar{y}\right)^{2}}\,,\qquad \mathrm{EV}=1-\dfrac{\mathrm{Var}(r)}{\mathrm{Var}(y)}\,,\qquad R^{2}=\mathrm{EV}-\dfrac{\bar{r}^{\,2}}{\mathrm{Var}(y)}$"],
'e5743': [r"$\rho_\alpha(r)=\max\left(\alpha\,r,\ (\alpha-1)\,r\right)\,,\qquad D^{2}=1-\dfrac{\sum_i d(y_i,\hat{y}_i)}{\sum_i d(y_i,\,y_0)}$"],
'e5744': [r"$d(y,\mu)=2\left[\dfrac{y^{\,2-q}}{(1-q)(2-q)}-\dfrac{y\,\mu^{\,1-q}}{1-q}+\dfrac{\mu^{\,2-q}}{2-q}\right]$",
          r"$q=0:\ (y-\mu)^{2}\,,\qquad q=1:\ 2\left(y\ln\dfrac{y}{\mu}-y+\mu\right)\,,\qquad q=2:\ 2\left(\ln\dfrac{\mu}{y}+\dfrac{y}{\mu}-1\right)$"],
}
# each line is rendered alone and the lines are stacked with a fixed gap
# (mkeq's own line spacing is too tight for fractions)
from PIL import Image
import tempfile
def stack(name, lines, gap=26):
    tmp = tempfile.mkdtemp(); ims = []
    for j, t in enumerate(lines):
        mkeq({'l%d' % j: [t]}, tmp); ims.append(Image.open('%s/l%d.png' % (tmp, j)).convert('RGB'))
    W = max(i.size[0] for i in ims); Ht = sum(i.size[1] for i in ims) + gap * (len(ims) - 1)
    out = Image.new('RGB', (W, Ht), 'white'); y = 0
    for i in ims:
        out.paste(i, ((W - i.size[0]) // 2, y)); y += i.size[1] + gap
    out.save(name + '.png', dpi=(300, 300))
for k, v in E.items(): stack(k, v)
