# Figures of Chapter 22, part 1: the eight seeds. Run: python3 fig/figs1.py
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
import matplotlib.text as mtext
from sklearn.svm import SVC
plt.rcParams.update({'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.labelsize':9,'legend.fontsize':8.5})
NUM={}; CHK={}
def sv(fig,name,printed_px):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image; w=Image.open(name).size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
X=np.array([[1,5],[2,3],[3,4],[4,1],[7,6],[8,2],[4,7],[6,3]],float); y=np.array([-1,-1,-1,-1,1,1,1,1])
w0=np.array([0.8,0.4]); b0=-5.0; nw=np.linalg.norm(w0); u=w0/nw
SVI=[2,6,7]; XL=(0,9.5); YL=(-0.2,8.4)
def pts(ax,X=X,y=y,s=34,alpha=1.0,zorder=5):
    for cls,col,mk in ((-1,BLUE,'o'),(1,OR,'s')):
        m=y==cls; ax.scatter(X[m,0],X[m,1],s=s,color=col,marker=mk,edgecolor='white',lw=0.8,zorder=zorder,alpha=alpha)
def rings(ax,P,s=120,col=NAVY): ax.scatter(P[:,0],P[:,1],s=s,facecolor='none',edgecolor=col,lw=1.2,zorder=6)
def ln(ax,w,b,c,**k):
    x1=np.array(XL); x2=(c-b-w[0]*x1)/w[1]; return ax.plot(x1,x2,**k)
def street(ax,w,b,fc='#E3F4EA',main=GREEN,lw=2.0,ls='-',alpha=1.0,z=2):
    xx=np.array(XL); ax.fill_between(xx,(-1-b-w[0]*xx)/w[1],(1-b-w[0]*xx)/w[1],color=fc,zorder=0,alpha=alpha)
    ln(ax,w,b,0,color=main,lw=lw,ls=ls,zorder=z); ln(ax,w,b,1,color=main,lw=1.0,ls='--',zorder=z); ln(ax,w,b,-1,color=main,lw=1.0,ls='--',zorder=z)
def axes(ax,lab=True):
    ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_aspect('equal')
    if lab: ax.set_xlabel('seed length  $x_1$ (mm)'); ax.set_ylabel('seed width  $x_2$ (mm)')
m_full=SVC(kernel='linear',C=1e6,tol=1e-10).fit(X,y)
NUM['full']={'w':[round(float(v),4) for v in m_full.coef_[0]],'b':round(float(m_full.intercept_[0]),4),'sv':[int(v)+1 for v in m_full.support_],'margin':round(float(2/np.linalg.norm(m_full.coef_)),4)}
# ---- f220 opener
fig,ax=plt.subplots(figsize=(4.4,3.05),dpi=240)
street(ax,w0,b0); pts(ax); rings(ax,X[SVI])
c0=np.array([3.0,4.0]); c1=c0+(2/nw)*u
ax.annotate('',xy=c1,xytext=c0,arrowprops=dict(arrowstyle='<->',color=RED,lw=1.3,shrinkA=0,shrinkB=0),zorder=7)
ax.text(5.45,4.95,'margin\n$2/\\Vert\\mathbf{w}\\Vert=2.24$',color=RED,fontsize=9,ha='left',va='center',zorder=8)
ax.text(0.25,0.45,'kind A  ($y=-1$)',color=BLUE,fontsize=9,weight='bold',va='center'); ax.text(9.4,3.8,'kind B\n($y=+1$)',color='#B8730A',fontsize=9,weight='bold',va='center',ha='right')
ax.annotate('support vectors',xy=(4.15,7.12),xytext=(5.6,8.0),fontsize=9,color=NAVY,ha='left',va='center',arrowprops=dict(arrowstyle='-',color=NAVY,lw=0.8))
for c,t in ((-1,'$f=-1$'),(0,'$f=0$'),(1,'$f=+1$')): ax.text((c-b0-0.4*1.0)/0.8+0.6,1.0,t,color=GREEN,fontsize=8.5,rotation=-63.4,ha='center',va='center')
axes(ax); sv(fig,'f220.png',440)
# ---- f221 many lines vs widest street
rng=np.random.default_rng(5); cands=[]
def separates(w,b): z=y*(X@w+b); return z.min()>0
for ang in np.deg2rad(np.arange(-80,81,4)):
    for _ in range(1):
        w=np.array([np.cos(ang),np.sin(ang)]); 
        for sgn in (1,-1):
            ww=sgn*w; zz=X@ww; lo=zz[y==-1].max(); hi=zz[y==1].min()
            if hi>lo: cands.append((ww,-(lo+hi)/2,(hi-lo)/2))
cands=sorted(cands,key=lambda t:t[2])
NUM['n_sep_dirs']=len(cands)
pick=[cands[0],cands[len(cands)//3],cands[len(cands)//2]] 
fig,axs=plt.subplots(1,2,figsize=(4.9,2.55),dpi=240)
ax=axs[0]; cols=[PUR,GREY,TEAL]; lss=['-','--',':']
for (w,b,h),c,l in zip(pick,cols,lss): ln(ax,w,b,0,color=c,lw=1.6,ls=l)
NUM['three_lines']=[{'w':[round(float(v),3) for v in w],'b':round(float(b),3),'half_margin':round(float(h),3)} for w,b,h in pick]
pts(ax); axes(ax,False); ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.set_title('many lines separate the kinds',fontsize=9.5,color=NAVY)
ax=axs[1]; street(ax,w0,b0); pts(ax); rings(ax,X[SVI]); axes(ax,False); ax.set_xlabel('$x_1$'); ax.set_yticklabels([]); ax.set_title('the widest street',fontsize=9.5,color=NAVY)
fig.tight_layout(w_pad=0.8); sv(fig,'f221.png',470)
# ---- f222 delete a non-support vector / a support vector
fig,axs=plt.subplots(1,2,figsize=(4.9,2.55),dpi=240)
for ax,drop,ttl in ((axs[0],4,'seed 5 removed: the street stays'),(axs[1],7,'seed 8 removed: the street moves')):
    keep=np.arange(8)!=drop; m=SVC(kernel='linear',C=1e6,tol=1e-10).fit(X[keep],y[keep]); w=m.coef_[0]; b=float(m.intercept_[0])
    street(ax,w0,b0,fc='none',main=GREY,lw=1.0,ls=':',z=1); street(ax,w,b,fc='#E3F4EA'); pts(ax,X[keep],y[keep]); rings(ax,m.support_vectors_)
    ax.scatter(*X[drop],s=60,marker='x',color=RED,lw=1.6,zorder=7); axes(ax,False); ax.set_xlabel('$x_1$'); ax.set_title(ttl,fontsize=9,color=NAVY)
    NUM['drop%d'%(drop+1)]={'w':[round(float(v),4) for v in w],'b':round(b,4),'margin':round(float(2/np.linalg.norm(w)),4),'sv':[int(np.arange(8)[keep][k])+1 for k in m.support_]}
axs[1].set_yticklabels([]); axs[0].set_ylabel('$x_2$')
fig.tight_layout(w_pad=0.8); sv(fig,'f222.png',470)
# ---- f231 geometry of the margin
fig,ax=plt.subplots(figsize=(4.4,3.0),dpi=240)
street(ax,w0,b0); pts(ax); rings(ax,X[SVI])
P=X[4]; f=w0@P+b0; foot=P-(f/nw**2)*w0
ax.plot([P[0],foot[0]],[P[1],foot[1]],color=RED,lw=1.4,zorder=6); ax.scatter(*foot,s=14,color=RED,zorder=7)
ax.text(6.1,4.2,'$d=f/\\Vert\\mathbf{w}\\Vert$\n$=3/0.894=3.35$',color=RED,fontsize=9,ha='left',va='center',zorder=8)
o=np.array([6.2,0.4]); o=o-((w0@o+b0)/nw**2)*w0   # point on boundary
arr(ax,tuple(o),tuple(o+1.5*u),c=PUR,lw=1.8,ms=9); ax.text(o[0]+1.5*u[0]+0.12,o[1]+1.5*u[1]-0.1,'$\\mathbf{w}$',color=PUR,fontsize=11,va='top')
axes(ax); sv(fig,'f231.png',420)
NUM['geom']={'f_seed5':float(f),'d_seed5':round(float(f/nw),4),'norm_w':round(float(nw),4),'margin':round(float(2/nw),4)}
json.dump(NUM,open('verify/fig_numbers1.json','w'),indent=1); json.dump(CHK,open('verify/fig_textsize1.json','w'),indent=1); print(json.dumps(NUM)); print(CHK)
