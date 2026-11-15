"""Shared figure style for the book (import with sys.path.insert(0,'<root>/lib')).
All text in STIX (Times-like serif) to match the Times New Roman body. Colours = book theme."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Polygon, Circle, FancyBboxPatch
import numpy as np
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':10,
  'axes.edgecolor':'#555555','axes.linewidth':0.7,'xtick.color':'#555555','ytick.color':'#555555',
  'axes.labelcolor':'#222222','legend.frameon':False,'axes.spines.top':False,'axes.spines.right':False})
NAVY='#0B1F3A'; BLUE='#1F5FBF'; LB='#E3EEFB'; OR='#F39C12'; RED='#C0392B'; PINK='#FBE6E3'; GREY='#555555'
GRID='#D0D5DD'; GREEN='#2E8B57'; PUR='#7B3FA0'; TEAL='#0E7C86'; MAG='#D6336C'; LGREY='#E9EDF3'
def arr(ax,p,q,c=RED,lw=1.0,ms=6,**k): ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=ms,color=c,lw=lw,**k))
def new(w=2.6,h=2.2,dpi=240):
    fig,ax=plt.subplots(figsize=(w,h),dpi=dpi); ax.set_aspect('equal'); ax.axis('off'); return fig,ax
def save(fig,n,dpi=240): fig.savefig(n,dpi=dpi,bbox_inches='tight',pad_inches=0.04,facecolor='white'); plt.close(fig)
def hatch_v(ax,x,y0,y1,step=0.6,d=0.35):
    ax.plot([x,x],[y0,y1],color=NAVY,lw=1)
    for y in np.arange(y0,y1+1e-9,step): ax.plot([x,x+d],[y,y+d],color=NAVY,lw=0.7)
def hatch_h(ax,x0,x1,y,step=0.6,d=0.35):
    ax.plot([x0,x1],[y,y],color=NAVY,lw=1)
    for x in np.arange(x0,x1+1e-9,step): ax.plot([x,x-d],[y,y-d],color=NAVY,lw=0.7)
def box(ax,x,y,w,h,txt,fc=LB,ec=BLUE,fs=9,tc=NAVY,bold=False,r=0.08):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0.02,rounding_size={r}',fc=fc,ec=ec,lw=0.9))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,color=tc,weight='bold' if bold else 'normal')
