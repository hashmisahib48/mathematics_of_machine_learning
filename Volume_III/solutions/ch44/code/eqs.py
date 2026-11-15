# Equation images of the Chapter 44 solutions. Run: python3 code/eqs.py  (writes sol/ch44/s44*.png)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from mkeq import mkeq
E = {
's44a3_4': r'$\mathrm{Perp}=e^{0.8018}=2.2296$',
's44a3_3': r'$H=-(0.7\ln0.7+0.2\ln0.2+0.1\ln0.1)=0.8018\ \mathrm{nats}$',
's44a2_8': r'$\mathbf{w}\;=\;\dfrac{(15.15,\;10.15)}{25.3}\;=\;(0.5988,\;0.4012)$',
's44a2_7': r'$\mathbf{w}\;\propto\;(10.15+5,\;5+5.15)=(15.15,\;10.15)$',
's44a2_5': r'$\mathbf{w}^{T}\mathbf{G}\,\mathbf{w}=0.36\times5-2\times0.24\times5+0.16\times10=1.8-2.4+1.6=1$',
's44a1_7': r'$\mathbf{z}^{(2)}=3.8485\times(0.7958,\,-0.5562,\,-0.2396)=(3.0624,\,-2.1404,\,-0.9220)$',
's44a1_6': r'$\mathbf{z}^{(1)}=7.2012\times(-0.1828,\,-0.5978,\,0.7806)=(-1.3163,\,-4.3046,\,5.6209)$',
's44a1_5': r'$\lambda=\dfrac{66.6667\pm37.0465}{2}=51.8566\,,\;\;14.8101$',
's44a1_10': r'$b_{12}=-\dfrac{1}{2}\,(36-33.3333-45.3333+44.4444)=-\dfrac{8}{9}$',
's44a1_3': r'$b_{11}=-\dfrac{1}{2}\,(0-33.3333-33.3333+44.4444)=\dfrac{100}{9}$',
's44a1_9': r'$\bar{g}=\dfrac{36+64+36+100+64+100}{9}=\dfrac{400}{9}=44.4444$',
's44a1_2': r'$\bar{r}=\left(\dfrac{100}{3},\,\dfrac{136}{3},\,\dfrac{164}{3}\right)=(33.3333,\,45.3333,\,54.6667)$',
# ---- 44A.1
's44a1_1': r'$b_{ij}\;=\;-\dfrac{1}{2}\left(d_{ij}^{2}-\bar{r}_{i}-\bar{r}_{j}+\bar{g}\right)\,,\qquad\bar{r}_{i}=\mathrm{row\ mean}\,,\quad\bar{g}=\mathrm{overall\ mean}$',
's44a1_4': r'$\lambda^{2}-\mathrm{tr}(\mathbf{B})\,\lambda+c_{2}=0\,,\qquad\mathrm{tr}(\mathbf{B})=\dfrac{600}{9}=66.6667\,,\quad c_{2}=768$',
# ---- 44A.2
's44a2_1': r'$\mathbf{c}_{1}=(0,0)-(2,1)=(-2,-1)\,,\qquad\mathbf{c}_{2}=(5,0)-(2,1)=(3,-1)$',
's44a2_2': r'$G_{11}=4+1=5\,,\qquad G_{12}=-6+1=-5\,,\qquad G_{22}=9+1=10$',
's44a2_4': r'$\mathbf{G}^{-1}\mathbf{1}\;=\;\dfrac{1}{25}\,(15,\;10)\;=\;(0.6,\;0.4)\,,\qquad\mathbf{w}=(0.6,\;0.4)$',
# ---- 44A.3
's44a3_1': r'$p=\left(\dfrac{1}{2},\dfrac{1}{4},\dfrac{1}{4}\right):\quad H=\dfrac{1}{2}\cdot1+\dfrac{1}{4}\cdot2+\dfrac{1}{4}\cdot2=1.5\ \mathrm{bits}\,,\qquad2^{1.5}=2.8284$',
's44a3_2': r'$p=\left(\dfrac{1}{4},\dfrac{1}{4},\dfrac{1}{4},\dfrac{1}{4}\right):\quad H=4\cdot\dfrac{1}{4}\cdot2=2\ \mathrm{bits}\,,\qquad2^{2}=4$',
# ---- 44A.4
's44a4_1': r'$\Vert z_1-z_2\Vert=3\,,\qquad\Vert z_1-z_3\Vert=7\,,\qquad\Vert z_2-z_3\Vert=4$',
's44a4_2': r'$\mathrm{stress}\;=\;(3-3)^{2}+(4-7)^{2}+(5-4)^{2}\;=\;0+9+1\;=\;10$',
's44a4_3': r'$\mathrm{Stress}\text{-}1\;=\;\sqrt{\dfrac{10}{3^{2}+7^{2}+4^{2}}}\;=\;\sqrt{\dfrac{10}{74}}\;=\;0.3676$',
# ---- questions
's44q3_1': r'$\Phi(\mathbf{Z})\;=\;\sum_{i}\Vert\mathbf{z}_{i}-\sum_{j}W_{ij}\mathbf{z}_{j}\Vert^{2}\;+\;\mu\sum_{(i,j)\in\mathrm{graph}}\left(\Vert\mathbf{z}_{i}-\mathbf{z}_{j}\Vert-d_{ij}\right)^{2}$',
}
mkeq(E, '.')
from PIL import Image
for k in E:
    w = Image.open(k + '.png').size[0]
    if w > 1450: print('too wide', k, w)
