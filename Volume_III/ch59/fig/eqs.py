# Equation images of Chapter 59. Run: python3 fig/eqs.py
import sys,os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from mkeq import mkeq
E={
'e5941': [r"$p\,c_{\mathrm{FN}}\;>\;(1-p)\,c_{\mathrm{FP}}\quad\Longleftrightarrow\quad p\;>\;t^{*}=\dfrac{c_{\mathrm{FP}}}{c_{\mathrm{FP}}+c_{\mathrm{FN}}}$"],
'e5971': [r"$\hat{a}\;\pm\;1.96\,\sqrt{\dfrac{\hat{a}\,(1-\hat{a})}{n}}\,,\qquad \hat{a}=\dfrac{\mathrm{correct}}{n}$"],
'e5972': [r"$\mathrm{fits}\;=\;K_{\mathrm{out}}\left(N_{\mathrm{cand}}\,K_{\mathrm{in}}+1\right)$"],
}
mkeq(E,'.')
