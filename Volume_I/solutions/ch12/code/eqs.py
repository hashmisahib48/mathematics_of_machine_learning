# Equation images of the Chapter 12 solutions. Run: python3 code/eqs.py  (writes sol/ch12/s12*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
# ---- 12A.1
's12a1_1': r'$S_{xx} \;=\; 4+1+1+4 \;=\; 10\,,\qquad S_{xy} \;=\; 6+1+2+4 \;=\; 13$',
's12a1_2': r'$\lambda+\alpha S_{xx} \;=\; 4+2\times 10 \;=\; 24\,,\qquad \sigma^{2} \;=\; \dfrac{1}{24}\,,\qquad \sigma \;=\; 0.2041$',
's12a1_3': r'$m \;=\; \dfrac{\alpha S_{xy}}{\lambda+\alpha S_{xx}} \;=\; \dfrac{2\times 13}{24} \;=\; \dfrac{13}{12} \;=\; 1.0833$',
's12a1_4': r'$\dfrac{\lambda}{\alpha} \;=\; \dfrac{4}{2} \;=\; 2\,,\qquad \dfrac{S_{xy}}{S_{xx}+2} \;=\; \dfrac{13}{12} \;=\; 1.0833$',
's12a1_5': r'$\dfrac{m}{\hat{w}_{\mathrm{OLS}}} \;=\; \dfrac{13/12}{13/10} \;=\; \dfrac{10}{12} \;=\; 0.8333$',
# ---- 12A.2
's12a2_1': r'$\sigma_{*}^{2} \;=\; \dfrac{1}{\alpha}+\dfrac{x_{*}^{2}}{\lambda+\alpha S_{xx}} \;=\; 1+\dfrac{x_{*}^{2}}{25}$',
's12a2_2': r'$1+\dfrac{x_{*}^{2}}{25} \;=\; 2^{2} \;\Longrightarrow\; x_{*}^{2} \;=\; 75 \;\Longrightarrow\; x_{*} \;=\; \pm\sqrt{75} \;=\; \pm 8.66$',
's12a2_3': r'$\sigma_{*}^{2} \;=\; 1+\dfrac{3^{2}}{25} \;=\; 1.36\,,\qquad \sigma_{*} \;=\; \sqrt{1.36} \;=\; 1.166$',
# ---- 12A.3
's12a3_1': r'$\lambda+\alpha S_{xx} \;=\; 1+2\times 20 \;=\; 41\,,\qquad m \;=\; \dfrac{\alpha S_{xy}}{41} \;=\; \dfrac{36}{41} \;=\; 0.8780$',
's12a3_2': r'$\gamma \;=\; \dfrac{\alpha S_{xx}}{\lambda+\alpha S_{xx}} \;=\; \dfrac{40}{41} \;=\; 0.9756$',
's12a3_3': r'$\lambda_{\mathrm{new}} \;=\; \dfrac{\gamma}{m^{2}} \;=\; \dfrac{0.9756}{0.7710} \;=\; 1.265$',
's12a3_4': r'$\alpha_{\mathrm{new}} \;=\; \dfrac{n-\gamma}{\mathrm{SSE}} \;=\; \dfrac{4-0.9756}{1.8096} \;=\; \dfrac{3.0244}{1.8096} \;=\; 1.671$',
# ---- 12B
's12b1_1': r'$\sigma_{w}^{2} \;=\; \dfrac{1}{\lambda+k\,\alpha S_{xx}}\;\approx\;\dfrac{1}{k\,\alpha S_{xx}}$',
's12b1_2': r'$\sigma_{*}^{2} \;=\; \dfrac{1}{\alpha}+x_{*}^{2}\,\sigma_{w}^{2} \;=\; 0.6000+36\times 0.0289 \;=\; 1.640$',
's12b2_1': r'$w_{1}^{2}+w_{2}^{2} \;\geq\; \dfrac{(w_{1}+w_{2})^{2}}{2}\,,\qquad \mathrm{equality\ for}\ w_{1}=w_{2}$',
# ---- 12C
's12c1_1': r'$\alpha_{\infty} \;=\; \dfrac{n}{\mathrm{SSE}_{\mathrm{OLS}}} \;=\; \dfrac{4}{1.8} \;=\; 2.222\,,\qquad \dfrac{1}{\sqrt{\alpha_{\infty}}} \;=\; 0.671$',
's12c1_2': r'$\sigma_{*}^{2}(x_{*}=0) \;=\; \dfrac{1}{\alpha}+0 \;=\; \dfrac{1}{\alpha}$',
's12c1_3': r'$\sigma_{\mathrm{mean\ of\ }m}^{2} \;=\; \dfrac{1}{m\,\alpha}$',
}
mkeq(E, '.'); print('solution equations ok:', len(E))
