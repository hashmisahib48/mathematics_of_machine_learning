# Equation images of the Chapter 4 solutions. Run: python3 code/eqs.py  (writes sol/ch04/s4*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 4A.1
's4a1_1': r"$P(R_{1}\cap R_{2})=P(R_{2}\mid R_{1})\,P(R_{1})=\dfrac{2}{4}\times\dfrac{3}{5}=0.3$",
's4a1_2': r"$P(R_{2}\mid R_{1})=\dfrac{P(R_{1}\cap R_{2})}{P(R_{1})}=\dfrac{0.3}{0.6}=0.5$",
's4a1_3': r"$P(R_{2})=P(R_{2}\mid R_{1})\,P(R_{1})+P(R_{2}\mid B_{1})\,P(B_{1})$",
's4a1_4': r"$P(R_{2})=0.5\times 0.6+0.75\times 0.4=0.3+0.3=0.6$",
# ---- 4A.2
's4a2_1': r"$P(+)=P(+\mid D)\,P(D)+P(+\mid D^{c})\,P(D^{c})$",
's4a2_3': r"$P(+)=0.95\times 0.02+0.10\times 0.98=0.019+0.098=0.117$",
's4a2_2': r"$P(D\mid +)=\dfrac{P(+\mid D)\,P(D)}{P(+)}=\dfrac{0.019}{0.117}=0.1624$",
# ---- 4A.3
's4a3_1': r"$\bar{x}=\dfrac{1+2+3+4}{4}=2.5\,,\qquad \bar{y}=\dfrac{2+1+4+3}{4}=2.5$",
's4a3_2': r"$S_{xy}=4\times 0.75=3$",
's4a3_5': r"$S_{xx}=2.25+0.25+0.25+2.25=5\,,\qquad S_{yy}=0.25+2.25+2.25+0.25=5$",
's4a3_3': r"$\mathrm{Cov}(x,y)=\dfrac{S_{xy}}{n-1}=\dfrac{3}{3}=1\,,\qquad \mathrm{Var}(x)=\mathrm{Var}(y)=\dfrac{5}{3}=1.6667$",
's4a3_4': r"$r=\dfrac{\mathrm{Cov}(x,y)}{\sqrt{\mathrm{Var}(x)\,\mathrm{Var}(y)}}=\dfrac{1}{1.6667}=0.6$",
# ---- 4A.4
's4a4_1': r"$z=\dfrac{180-170}{8}=1.25$",
's4a4_3': r"$P(X>180)=1-\Phi(1.25)=1-0.8944=0.1056$",
's4a4_2': r"$x_{0.9}=\mu+z_{0.9}\,\sigma=170+1.2816\times 8=180.25\ \mathrm{cm}$",
# ---- 4B
's4b1_1': r"$P(N=2)=\dfrac{\lambda^{2}e^{-\lambda}}{2!}=\dfrac{9\,e^{-3}}{2}=0.2240$",
's4b1_2': r"$P(N\geq 1)=1-P(N=0)=1-e^{-3}=0.9502$",
's4b2_1': r"$-\ln L(\mu,\sigma^{2})=\dfrac{n}{2}\ln(2\pi\sigma^{2})+\dfrac{1}{2\sigma^{2}}\sum_{i}(x_{i}-\mu)^{2}$",
's4b2_2': r"$\hat{\mu}=\bar{x}=\dfrac{40}{5}=8\,,\qquad \hat{\sigma}^{2}=\dfrac{18}{5}=3.6\,,\qquad \dfrac{18}{4}=4.5$",
's4b3_1': r"$J(m)=\sum_{i}(y_{i}-m)^{2}+\alpha\,m^{2}$",
's4b3_2': r"$J'(m)=-2\sum_{i}(y_{i}-m)+2\alpha m=0\ \ \Longrightarrow\ \ m=\dfrac{\sum_{i}y_{i}}{n+\alpha}$",
# ---- 4C
's4c1_1': r"$\mathrm{Cov}(x,x^{2})=E[x\cdot x^{2}]-E[x]\,E[x^{2}]=E[x^{3}]-E[x]\,E[x^{2}]=0-0=0$",
's4c2_1': r"$\dfrac{P(D\mid +)}{P(D^{c}\mid +)}=\dfrac{P(D)}{P(D^{c})}\times\dfrac{P(+\mid D)}{P(+\mid D^{c})}=\dfrac{P(D)}{P(D^{c})}\times\dfrac{0.90}{0.05}$",
's4c2_2': r"$\mathrm{odds}\ \to\ \mathrm{odds}\times 18\,,\qquad \mathrm{odds}\times 18^{2}=\mathrm{odds}\times 324$",
's4c2_3': r"$0.0101\times 18=0.1818\ \ (P=0.1538)$",
's4c2_4': r"$0.0101\times 324=3.2727\ \ (P=0.7660)$",
's4c3_1': r"$m=\dfrac{n\,\bar{x}+\alpha\,\mu_{0}}{n+\alpha}=w\,\bar{x}+(1-w)\,\mu_{0}\,,\qquad w=\dfrac{n}{n+\alpha}\,,\quad \alpha=\dfrac{\sigma^{2}}{\tau^{2}}$",
# ---- questions
's4q5_1': r"$\mu_{n}=\dfrac{n\bar{x}/\sigma^{2}+\mu_{0}/\tau^{2}}{n/\sigma^{2}+1/\tau^{2}}\ \ \to\ \ \bar{x}\qquad(\tau^{2}\to\infty)$",
's4q5_2': r"$\tau_{n}^{2}=\dfrac{1}{n/\sigma^{2}+1/\tau^{2}}\ \ \to\ \ \dfrac{\sigma^{2}}{n}\qquad(\tau^{2}\to\infty)$",
}
mkeq(E, '.')
print('solution equations ok:', len(E))
