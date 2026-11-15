# Equation images of the Chapter 47 solutions. Run: python3 code/eqs.py  (writes sol/ch47/s47*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 47A.1
's47a1_1': r'$u_i \;=\; \dfrac{|3-x_i|}{h} \;=\; \dfrac{|3-x_i|}{2.5} \;=\; 0.8\,,\ \ 0.4\,,\ \ 0.4\,,\ \ 2.4$',
's47a1_2': r'$\hat{f}(3) \;=\; \dfrac{0.5+0.5+0.5+0}{4\times 2.5} \;=\; \dfrac{1.5}{10} \;=\; 0.150$',
's47a1_3': [r'$K(u)=0.75\,(1-u^{2})\ \ \mathrm{for}\ u<1\,,\ \ \mathrm{else}\ 0$', r'$K=0.75\times0.36\,,\ 0.75\times0.84\,,\ 0.75\times0.84\,,\ 0\;=\;0.27\,,\ 0.63\,,\ 0.63\,,\ 0$'],
's47a1_4': r'$\hat{f}(3) \;=\; \dfrac{0.27+0.63+0.63+0}{4\times 2.5} \;=\; \dfrac{1.53}{10} \;=\; 0.153$',
# ---- 47A.2
's47a2_1': r'$\dfrac{h_{\mathrm{Scott}}}{\sigma}=n^{-1/(p+4)}\,,\qquad \dfrac{h_{\mathrm{Silverman}}}{\sigma}=\left(\dfrac{n\,(p+2)}{4}\right)^{-1/(p+4)}$',
's47a2_2': [r'$p=1:\quad \mathrm{Scott}\ \ 100^{-1/5}=e^{-4.6052/5}=e^{-0.9210}=0.3981$', r'$\qquad\qquad\ \mathrm{Silverman}\ \ 75^{-1/5}=e^{-4.3175/5}=e^{-0.8635}=0.4217$'],
's47a2_3': r'$\dfrac{n\,(p+2)}{4}=n \;\Longleftrightarrow\; p+2=4 \;\Longleftrightarrow\; p=2$',
# ---- 47A.3
's47a3_1': [r'$\varphi(u)=e^{-u^{2}/2}/\sqrt{2\pi}\,:\ \ \varphi(0)=0.39894\,,\ \varphi(1)=0.24197\,,$', r'$\varphi(7)\approx 9\times10^{-12}\,,\ \varphi(8)\approx 5\times10^{-15}$'],
's47a3_2': r'$u=\dfrac{52-x_j}{4}=\dfrac{52-56}{4}\,,\ \dfrac{52-80}{4}\,,\ \dfrac{52-84}{4}\;=\;-1\,,\ -7\,,\ -8$',
's47a3_3': [r'$\hat{f}_{-1}(52) \;=\; \dfrac{\varphi(1)+\varphi(7)+\varphi(8)}{(n-1)\,h} \;=\; \dfrac{0.24197}{3\times4} \;=\; 0.02016$', r'$\ln\hat{f}_{-1}(52)=-3.9038$'],
's47a3_4': [r'$\hat{f}(52) \;=\; \dfrac{\varphi(0)+\varphi(1)+\varphi(7)+\varphi(8)}{n\,h} \;=\; \dfrac{0.64091}{16} \;=\; 0.04006$', r'$\ln\hat{f}(52)=-3.2175$'],
's47a3_5': [r'$\hat{f}_{-1}(52) \;=\; \dfrac{n\,\hat{f}(52)-K(0)/h}{n-1}$', r'$\;=\; \dfrac{4\times0.04006-0.39894/4}{3} \;=\; \dfrac{0.16023-0.09974}{3} \;=\; 0.02016$'],
's47a3_6': r'$\overline{\ln\hat{f}}-\overline{\ln\hat{f}_{-}} \;=\; -3.2175-(-3.9038) \;=\; 0.6864$',
# ---- questions
's47q2_1': r'$\dfrac{1}{n}\sum_{i}\ln\hat{f}(x_i)\;\approx\;-\ln\!\left(n\,h\sqrt{2\pi}\right)\;\to\;+\infty\qquad(h\to0)$',
's47q2_2': r'$\ln\hat{f}(x)\;\approx\;-\dfrac{d^{2}}{2h^{2}}-\ln\!\left(n\,h\sqrt{2\pi}\right)\;\to\;-\infty\,,\qquad d=\min_i|x-x_i|>0$',
's47q3_1': r'$R(f^{\prime\prime})\;\approx\;\sum_{k} w_k^{2}\,\dfrac{3}{8\sqrt{\pi}\,\sigma_k^{5}}\qquad\mathrm{(well\ separated\ humps)}$',
}
mkeq(E, '.'); print('solution equations ok:', len(E))
