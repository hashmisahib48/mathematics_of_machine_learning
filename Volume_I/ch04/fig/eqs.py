# Equation images of Chapter 4. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq, matimg
E={
'e411': r'$P(A^{c}) \;=\; 1 - P(A)$',
'e412': r'$P(A \cup B) \;=\; P(A) + P(B) - P(A \cap B)$',
'e421': r'$P(A \mid B) \;=\; \dfrac{P(A \cap B)}{P(B)}$',
'e422': r'$P(B) \;=\; P(B \mid A)\,P(A) + P(B \mid A^{c})\,P(A^{c})$',
'e423': r'$P(A \mid B) \;=\; \dfrac{P(B \mid A)\,P(A)}{P(B)}$',
'e424': r'$P(A \cap B) \;=\; P(A)\,P(B)$',
'e425': r'$P(A \cap B \mid C) \;=\; P(A \mid C)\,P(B \mid C)$',
'e431': r'$P(X = k) \;=\; \binom{n}{k}\,\theta^{k}\,(1-\theta)^{n-k}\,, \qquad k = 0, 1, \ldots, n$',
'e432': r'$P(N = k) \;=\; \dfrac{\lambda^{k}\,e^{-\lambda}}{k!}\,, \qquad k = 0, 1, 2, \ldots$',
'e441': r'$P(a \leq X \leq b) \;=\; \int_{a}^{b} p(x)\,dx\,, \qquad \int_{-\infty}^{\infty} p(x)\,dx = 1$',
'e442': r'$p(x) \;=\; \dfrac{1}{\sqrt{2\pi\sigma^{2}}}\,\exp\!\left(-\dfrac{(x-\mu)^{2}}{2\sigma^{2}}\right)$',
'e443': r'$Z \;=\; \dfrac{X - \mu}{\sigma} \;\sim\; \mathcal{N}(0, 1)$',
'e444': r'$p(x) \;=\; \lambda\,e^{-\lambda x}\,, \qquad x \geq 0$',
'e445': r'$p(x) \;=\; \dfrac{\beta^{k}}{\Gamma(k)}\,x^{k-1}\,e^{-\beta x}\,, \qquad x > 0$',
'e451': r'$E[X] \;=\; \sum_{k} x_k\,P(X = x_k) \qquad \mathrm{or} \qquad E[X] \;=\; \int x\,p(x)\,dx$',
'e452': r'$\mathrm{Var}[X] \;=\; E\!\left[(X-\mu)^{2}\right] \;=\; E[X^{2}] - \mu^{2}\,, \qquad \sigma \;=\; \sqrt{\mathrm{Var}[X]}$',
'e453': r'$E[aX + b] \;=\; a\,E[X] + b\,, \qquad \mathrm{Var}[aX + b] \;=\; a^{2}\,\mathrm{Var}[X]$',
'e454': r'$E[X + Y] \;=\; E[X] + E[Y]\,, \qquad \mathrm{Var}[X + Y] \;=\; \mathrm{Var}[X] + \mathrm{Var}[Y] $',
'e461': r'$\mathrm{Cov}(X, Y) \;=\; E\!\left[(X-\mu_X)(Y-\mu_Y)\right] \;=\; E[XY] - \mu_X\,\mu_Y$',
'e462': r'$\rho \;=\; \dfrac{\mathrm{Cov}(X, Y)}{\sigma_X\,\sigma_Y}\,, \qquad -1 \leq \rho \leq 1$',
'e464': r'$p(\mathbf{x}) \;=\; \dfrac{1}{\sqrt{(2\pi)^{d}\,\mathrm{det}\,\Sigma}}\,\exp\!\left(-\frac{1}{2}\,(\mathbf{x}-\mu)^{\mathrm{T}}\,\Sigma^{-1}\,(\mathbf{x}-\mu)\right)$',
'e471': r'$\bar{x} \;=\; \dfrac{1}{n}\sum_{i=1}^{n} x_i\,, \qquad s^{2} \;=\; \dfrac{1}{n-1}\sum_{i=1}^{n}\,(x_i - \bar{x})^{2}$',
'e473': r'$\mathrm{SE}(\bar{x}) \;=\; \dfrac{s}{\sqrt{n}}$',
'e481': r'$L(\theta) \;=\; \prod_{i=1}^{n} p(x_i \mid \theta)\,, \qquad \ln L(\theta) \;=\; \sum_{i=1}^{n} \ln p(x_i \mid \theta)$',
'e482': r'$\ln L(\theta) \;=\; k\,\ln\theta + (n-k)\,\ln(1-\theta)$',
'e484': r'$\ln L(\mu, \sigma^{2}) \;=\; -\dfrac{n}{2}\ln(2\pi\sigma^{2}) \;-\; \dfrac{1}{2\sigma^{2}}\sum_{i=1}^{n}\,(x_i - \mu)^{2}$',
'e485': r'$\hat{\mu} \;=\; \dfrac{1}{n}\sum_{i=1}^{n} x_i \;=\; \bar{x}\,, \qquad \hat{\sigma}^{2} \;=\; \dfrac{1}{n}\sum_{i=1}^{n}\,(x_i - \bar{x})^{2}$',
'e491': r'$p(\theta \mid \mathrm{data}) \;=\; \dfrac{p(\mathrm{data} \mid \theta)\,p(\theta)}{p(\mathrm{data})} \;\propto\; L(\theta)\,p(\theta)$',
'e492': r'$\hat{\theta}_{\mathrm{MAP}} \;=\; \mathrm{arg\,max}_{\theta}\,\left[\,\ln L(\theta) + \ln p(\theta)\,\right]$',
'e493': r'$\theta \sim \mathrm{Beta}(a, b) \;\;\Rightarrow\;\; \theta \mid \mathrm{data} \;\sim\; \mathrm{Beta}(a + k,\; b + n - k)$',
'e494': r'$\mu_{\mathrm{post}} \;=\; \dfrac{n\bar{x}/\sigma^{2} + \mu_0/\tau^{2}}{n/\sigma^{2} + 1/\tau^{2}}\,, \qquad \sigma_{\mathrm{post}}^{2} \;=\; \dfrac{1}{n/\sigma^{2} + 1/\tau^{2}}$',
'e495': r'$\hat{\mathbf{w}}_{\mathrm{MAP}} \;=\; \mathrm{arg\,min}_{\mathbf{w}}\;\left[\,\sum_{i=1}^{n}\,(y_i - \mathbf{w}^{\mathrm{T}}\mathbf{x}_i)^{2} \;+\; \dfrac{\sigma^{2}}{\tau^{2}}\,\Vert\mathbf{w}\Vert^{2}\,\right]$',
'e4101': r'$\mathrm{MSE}(\hat{\theta}) \;=\; E\!\left[(\hat{\theta}-\theta)^{2}\right] \;=\; \left(E[\hat{\theta}]-\theta\right)^{2} + \mathrm{Var}[\hat{\theta}]$',
}
mkeq(E,'.')
import matplotlib.pyplot as plt
def two(name,l1,l2,ls=2.6):
    fig=plt.figure(figsize=(.01,.01)); fig.text(0,0,l1+'\n'+l2,fontsize=11.5,linespacing=ls); fig.savefig(name+'.png',dpi=300,bbox_inches='tight',pad_inches=0.02,facecolor='white'); plt.close(fig)
two('e472',r'$\sum_{i=1}^{n}(x_i-\mu)^{2} \;=\; \sum_{i=1}^{n}(x_i-\bar{x})^{2} + n\,(\bar{x}-\mu)^{2}$',r'$n\,\sigma^{2} \;=\; E\left[\sum_{i}(x_i-\bar{x})^{2}\right] + n\cdot\dfrac{\sigma^{2}}{n} \;\Rightarrow\; E\left[\sum_{i}(x_i-\bar{x})^{2}\right] = (n-1)\,\sigma^{2}$')
two('e486',r'$\dfrac{\partial \ln L}{\partial \mu} \;=\; \dfrac{1}{\sigma^{2}}\sum_{i=1}^{n}(x_i-\mu) \;=\; 0$',r'$\dfrac{\partial \ln L}{\partial \sigma^{2}} \;=\; -\dfrac{n}{2\sigma^{2}} + \dfrac{1}{2\sigma^{4}}\sum_{i=1}^{n}(x_i-\mu)^{2} \;=\; 0$')
M=lambda rows:[[str(v) for v in r] for r in rows]
matimg('m463',[r'$\Sigma \;=\;$',[[r'\sigma_1^{2}',r'\sigma_{12}'],[r'\sigma_{12}',r'\sigma_2^{2}']],r'$,\qquad \sigma_{12}=\mathrm{Cov}(X_1,X_2)=\rho\,\sigma_1\sigma_2$'],rh=0.26)
# derivation of the coin estimate, two lines
fig=plt.figure(figsize=(.01,.01))
fig.text(0,0,r'$\dfrac{d\,\ln L}{d\theta} \;=\; \dfrac{k}{\theta} - \dfrac{n-k}{1-\theta} \;=\; 0 \qquad\Longrightarrow\qquad \hat{\theta} \;=\; \dfrac{k}{n}$',fontsize=11.5)
fig.savefig('e483.png',dpi=300,bbox_inches='tight',pad_inches=0.02,facecolor='white'); plt.close(fig)
print('eqs ok')
