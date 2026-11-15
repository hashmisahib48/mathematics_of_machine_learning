# Figure 22.8-1: flow chart of SVC.fit. Run: python3 fig/figs3.py
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
fig,ax=new(3.8,5.6)
def fb(xc,yc,w,h,t,fc=LB,ec=BLUE,fs=8.3):
    ax.add_patch(FancyBboxPatch((xc-w/2,yc-h/2),w,h,boxstyle='round,pad=0.02,rounding_size=0.06',fc=fc,ec=ec,lw=0.9)); ax.text(xc,yc,t,ha='center',va='center',fontsize=fs,color=NAVY,linespacing=1.15)
A=lambda p,q,**k: arr(ax,p,q,c=NAVY,lw=0.9,ms=8,**k)
fb(3.4,8.3,6.4,0.55,'Input:  $\\mathbf{X}$,  labels $y$,  optional sample_weight',fc=LGREY,ec=GREY)
fb(3.4,7.3,6.4,0.8,'Find classes_  ($K$ labels); class_weight_;\n$C_i=C\\times$ class weight $\\times$ sample weight')
fb(3.4,6.2,6.4,0.8,'gamma = 1/($p\\cdot$ Var $\\mathbf{X}$)  if "scale"\n= 1/$p$  if "auto";  a number is used as given')
ax.add_patch(Polygon([(3.4,5.65),(4.8,5.2),(3.4,4.75),(2.0,5.2)],fc='#FDF0E0',ec=OR,lw=0.9)); ax.text(3.4,5.2,'$K=2$ ?',ha='center',va='center',fontsize=8.6,color=NAVY)
fb(1.2,4.0,2.2,0.85,'$K=2$:\none binary\nproblem',fc='#E3F4EA',ec=GREEN,fs=8); fb(5.6,4.0,2.2,0.85,'$K\\geq3$: one-vs-\none, $K(K-1)/2$\nproblems',fc='#F1EAF7',ec=PUR,fs=8)
fb(3.4,2.4,6.4,1.7,'For every binary problem, SMO on the dual (Eq. 22.4-6):\npick the most violating pair  $(i,j)$  (second order);\nsolve for $\\alpha_i,\\alpha_j$ in closed form, clip to $[0,C]$;\nupdate the gradient from two kernel rows (cached);\nshrink every min($n$, 1000) steps;  stop at gap < tol',fc=PINK,ec=RED,fs=8)
fb(3.4,1.05,6.4,0.7,'$b$ from the free support vectors;\nkeep samples with $\\alpha_i>0$',fs=8.3)
fb(3.4,0.0,6.4,0.7,'Store  support_, support_vectors_, dual_coef_,\nintercept_, n_support_;  return self',fc=LGREY,ec=GREY,fs=8.3)
A((3.4,8.02),(3.4,7.7)); A((3.4,6.9),(3.4,6.6)); A((3.4,5.8),(3.4,5.66))
ax.plot([2.0,1.2],[5.2,5.2],color=NAVY,lw=0.9); A((1.2,5.2),(1.2,4.45)); ax.plot([4.8,5.6],[5.2,5.2],color=NAVY,lw=0.9); A((5.6,5.2),(5.6,4.45))
for xs in (1.2,5.6): ax.plot([xs,xs],[3.575,3.42],color=NAVY,lw=0.9)
ax.plot([1.2,5.6],[3.42,3.42],color=NAVY,lw=0.9); A((3.4,3.42),(3.4,3.27))
A((3.4,1.55),(3.4,1.4)); A((3.4,0.7),(3.4,0.37))
ax.set_xlim(0,6.8); ax.set_ylim(-0.45,8.7)
fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip())
fig.savefig('f281.png',dpi=240,bbox_inches='tight',pad_inches=0.04,facecolor='white'); w=Image.open('f281.png').size[0]/240; print(w,Image.open('f281.png').size)
json.dump({'f281.png':{'fig_in':round(w,2),'min_font_design':fs,'min_font_printed':round(fs*(380/96)/w,1)}},open('verify/fig_textsize3.json','w'))
