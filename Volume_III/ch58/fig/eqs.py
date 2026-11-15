# Equation images of Chapter 58. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq
E={
'e5831': [r"$R(\mathrm{yes}\mid\mathbf{x})\;=\;(1-p)\,c_{FP}\,,\qquad R(\mathrm{no}\mid\mathbf{x})\;=\;p\,c_{FN}$", r"$\mathrm{say\ yes}\ \Longleftrightarrow\ p\,c_{FN}\;>\;(1-p)\,c_{FP}\,,\qquad p=P(y=1\mid\mathbf{x})$"],
'e5832': [r"$t^{*}\;=\;\dfrac{c_{FP}}{c_{FP}+c_{FN}}\;=\;\dfrac{1}{1+c_{FN}/c_{FP}}$", r"$t^{*}\;=\;\dfrac{c_{FP}-c_{TN}}{(c_{FP}-c_{TN})+(c_{FN}-c_{TP})}\qquad(\mathrm{costs\ of\ all\ four\ outcomes})$"],
'e5833': [r"$\ln\dfrac{t^{*}}{1-t^{*}}\;=\;\ln\dfrac{c_{FP}}{c_{FN}}$"],
'e5834': [r"$\hat{\tau}\;=\;\arg\max_{\tau}\;\dfrac{1}{K}\sum_{k=1}^{K}\tilde{s}_k(\tau)$", r"$\tilde{s}_k:\ \mathrm{scores}\ s_{km}\ \mathrm{of\ fold}\ k\ \mathrm{at\ its\ thresholds}\ \tau_{km},\ \mathrm{interpolated}$"],
'e5841': [r"$I_j\;=\;s(f;\mathbf{X},\mathbf{y})\;-\;\dfrac{1}{R}\sum_{r=1}^{R}s\left(f;\mathbf{X}^{(j,r)},\mathbf{y}\right)$"],
'e5842': [r"$\mathrm{E}\left[\mathrm{accuracy}\ \mathrm{after\ shuffling}\right]\;=\;\dfrac{n_{1}\,m_{1}+n_{0}\,m_{0}}{n^{2}}$"],
'e5843': [r"$\mathrm{PD}_j(v)\;=\;\dfrac{1}{n}\sum_{i=1}^{n}\mathrm{ICE}_i(v)\,,\qquad \mathrm{ICE}_i(v)\;=\;f\left(v,\ \mathbf{x}_{i,\setminus j}\right)$"],
}
mkeq(E,'.')
