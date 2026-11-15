# All figures of Chapter 54. Run: python3 fig/figs.py  (PNGs into ch54/, numbers into verify/fig_numbers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
plt.rcParams.update({'xtick.labelsize':9,'ytick.labelsize':9,'axes.labelsize':9,'legend.fontsize':9})
NUM={}; CHK={}
PX={'f540.png':340,'f542.png':290,'f545.png':310,'f571.png':290,'f5102.png':300,'f542b.png':270,'f5111.png':340}
def sv(fig,name,printed_px,target=7.7):
    printed_px=PX.get(name,printed_px)
    from PIL import Image
    for it in range(5):
        fig.canvas.draw()
        T=[t for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()]
        fs=min(t.get_fontsize() for t in T)
        fig.savefig(name,dpi=240,bbox_inches='tight',pad_inches=0.04,facecolor='white')
        with Image.open(name) as im: w=im.size[0]/240
        pr=fs*printed_px/96/w
        if pr>=target: break
        k=min(1.3,(target+0.05)/pr)
        for t in T: t.set_fontsize(t.get_fontsize()*k)
        try: fig.tight_layout(w_pad=1.0)
        except Exception: pass
    plt.close(fig)
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(pr,1)}
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
SW=json.load(open('verify/sweeps.json'))

# ======== 54.0-1 opener: leaky and honest order, and the scores they give
fig=plt.figure(figsize=(4.5,1.6),dpi=240)
ax=fig.add_axes([0.0,0.0,0.50,1.0]); ax.axis('off'); ax.set_xlim(-0.15,10.15); ax.set_ylim(0,8.4)
ax.text(0.0,7.9,'leaky: select first',fontsize=9.7,color=RED,weight='bold',va='center')
box(ax,0,5.2,3.0,1.9,'all\nrows',fs=9.7); box(ax,3.7,5.2,3.2,1.9,'select\nfeatures',fs=9.7,fc=PINK,ec=RED); box(ax,7.6,5.2,2.4,1.9,'5-fold\nCV',fs=9.7)
arr(ax,(3.05,6.15),(3.65,6.15),c=NAVY); arr(ax,(6.95,6.15),(7.55,6.15),c=NAVY)
ax.text(0.0,4.0,'honest: a pipeline',fontsize=9.7,color=GREEN,weight='bold',va='center')
box(ax,0,0.4,3.0,1.9,'training\nfolds',fs=9.7); box(ax,3.7,0.4,3.2,1.9,'select,\nfit',fs=9.7,fc='#E6F4EC',ec=GREEN); box(ax,7.6,0.4,2.4,1.9,'score\nheld-out',fs=9.7,fc=LGREY,ec=GREY)
arr(ax,(3.05,1.35),(3.65,1.35),c=NAVY); arr(ax,(6.95,1.35),(7.55,1.35),c=NAVY)
ax=fig.add_axes([0.63,0.27,0.36,0.70]); H=SW['raw']['hist']; bins=np.linspace(0.2,1.0,17)
ax.hist([b for a,b in H],bins=bins,color=GREEN,alpha=0.85,label='inside the pipeline')
ax.hist([a for a,b in H],bins=bins,color=RED,alpha=0.85,label='selected first',hatch='///',edgecolor='white',lw=0.3)
ax.axvline(0.5,color=GREY,lw=1.0,ls='--'); ax.set_xlabel('5-fold accuracy',fontsize=9.7); ax.set_ylabel('data sets',fontsize=9.7); ax.tick_params(labelsize=9.7); ax.set_xlim(0.2,1.0)
ax.text(0.485,31,'inside the\npipeline',fontsize=9.7,color=GREEN,va='center',ha='right',weight='bold')
ax.text(0.745,33,'selected\nfirst',fontsize=9.7,color=RED,va='center',ha='right',weight='bold')
ax.text(0.5,50,' chance',fontsize=9.7,color=GREY,va='center',ha='left')
ax.set_ylim(0,53)
sv(fig,'f540.png',360)
NUM['hist']=SW['summary']['hist']

# ======== 54.2-1 column routing for the cars
fig=plt.figure(figsize=(3.55,1.95),dpi=240); ax=fig.add_axes([0,0,1,1]); ax.axis('off'); ax.set_xlim(0,12.4); ax.set_ylim(-0.2,6.7)
cols=[('colour',BLUE),('fuel',BLUE),('trim',BLUE),('city',PUR),('age',GREEN),('km',GREEN)]
ys=[5.7,4.7,3.7,2.7,1.5,0.5]
for (nm,cc),y in zip(cols,ys):
    ax.add_patch(plt.Rectangle((0,y-0.4),1.9,0.8,fc='white',ec=cc,lw=1.2)); ax.text(0.95,y,nm,ha='center',va='center',fontsize=9,color=NAVY)
ax.text(0.95,6.45,'columns',ha='center',va='center',fontsize=9,color=GREY,weight='bold')
TB=[('OneHot\nEncoder',BLUE,LB,4.7,2.1),('Target\nEncoder',PUR,'#EFE6F6',2.7,1.3),('Standard\nScaler',GREEN,'#E6F4EC',1.0,1.4)]
for nm,cc,fc,y,h in TB:
    ax.add_patch(FancyBboxPatch((4.0,y-h/2),2.9,h,boxstyle='round,pad=0.02,rounding_size=0.1',fc=fc,ec=cc,lw=1.1)); ax.text(5.45,y,nm,ha='center',va='center',fontsize=9,color=NAVY)
for y in ys[:3]: arr(ax,(1.95,y),(3.95,4.7+(y-4.7)*0.45),c=GREY,lw=0.8)
arr(ax,(1.95,2.7),(3.95,2.7),c=GREY,lw=0.8)
for y in ys[4:]: arr(ax,(1.95,y),(3.95,1.0+(y-1.0)*0.5),c=GREY,lw=0.8)
ax.text(9.2,6.45,'output $\\mathbf{Z}$, side by side',ha='center',va='center',fontsize=9,color=GREY,weight='bold')
segs=[(14,BLUE),(1,PUR),(2,GREEN)]; yb=[4.7,2.7,1.0]; x0=8.0; W=3.3; tot=17; xs=x0
for (w,cc),yy in zip(segs,yb):
    ww=W*w/tot; ax.add_patch(plt.Rectangle((xs,yy-0.4),ww,0.8,fc=cc,ec=cc,alpha=0.55,lw=0.8))
    ax.text(xs+ww/2,yy+0.95,str(w),ha='center',va='center',fontsize=9,color=cc,weight='bold')
    arr(ax,(6.95,yy),(xs-0.05 if xs>x0 else x0-0.05,yy),c=GREY,lw=0.8); xs+=ww
ax.plot([x0,x0],[0.4,5.2],color=GRID,lw=0.7,ls=':'); ax.text(x0+W/2,-0.15,'17 columns',ha='center',va='center',fontsize=9,color=NAVY)
sv(fig,'f542.png',400)

# ======== 54.5-1 retransformation bias
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor as TTR
PN=json.load(open('verify/prose_numbers.json'))
fig,axs=plt.subplots(1,2,figsize=(4.7,1.85),dpi=240,gridspec_kw={'width_ratios':[1.25,1]})
rng=np.random.RandomState(3); n=250; x=rng.uniform(0,3,n); sig=0.6
y=np.exp(1+0.8*x+rng.normal(0,sig,n)); xs=np.linspace(0,3,100)
ax=axs[0]; ax.scatter(x,y,s=4,color=BLUE,alpha=0.45,lw=0)
ax.plot(xs,np.exp(1+0.8*xs),color=GREEN,lw=1.5,ls='--',label='median $e^{\\mu}$')
ax.plot(xs,np.exp(1+0.8*xs+sig**2/2),color=PUR,lw=1.5,label='mean $e^{\\mu+\\sigma^2/2}$')
ax.set_ylim(0,45); ax.set_xlim(0,3); ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.legend(loc='upper left',handlelength=1.8,borderaxespad=0.1); grid(ax)
ax=axs[1]; sg=np.linspace(0,1,50); ax.plot(sg,np.exp(sg**2/2),color=PUR,lw=1.5,label='$e^{\\sigma^2/2}$')
sk=sorted(PN['retransform'],key=float); ax.plot([float(k) for k in sk],[PN['retransform'][k]['ratio_means'] for k in sk],'o',color=RED,ms=4,label='measured')
ax.axhline(1,color=GREEN,lw=1.2,ls='--'); ax.set_xlabel('noise level $\\sigma$'); ax.set_ylabel('mean of $y$ / prediction'); ax.legend(loc='upper left',handlelength=1.4,borderaxespad=0.1); grid(ax); ax.set_ylim(0.95,1.5)
fig.tight_layout(w_pad=0.9); sv(fig,'f545.png',420)

# ======== 54.2-2 the hand example
X=np.array([[1,20],[3,50],[4,40],[6,70],[9,140],[13,100]],float); yv=np.array([0,0,0,1,1,1]); mu,sd=X.mean(0),X.std(0); Z=(X-mu)/sd; new=(np.array([8,80.])-mu)/sd
fig=plt.figure(figsize=(3.0,1.8),dpi=240); ax=fig.add_axes([0.13,0.2,0.55,0.77])
v=np.array([1,1])/np.sqrt(2); tt=np.array([-2.7,2.7])
ax.plot(tt*v[0],tt*v[1],color=GREEN,lw=1.4,label='PC1')
for z,c_ in zip(Z,yv): ax.scatter(*z,s=22,marker='o' if c_==0 else 's',color=BLUE if c_==0 else OR,zorder=3)
ax.scatter(*new,s=60,marker='*',color=RED,zorder=4)
p=(new@v)*v; ax.plot([new[0],p[0]],[new[1],p[1]],color=RED,lw=0.9,ls=':')
ax.text(new[0]+0.15,new[1]-0.45,'new',color=RED,fontsize=9)
ax.set_xlabel('scaled age $z_1$'); ax.set_ylabel('scaled km $z_2$'); ax.set_aspect('equal'); ax.set_xlim(-2.0,2.4); ax.set_ylim(-2.0,2.3); grid(ax)
ax.set_xticks([-2,0,2]); ax.set_yticks([-2,0,2])
ax.scatter([],[],s=22,marker='o',color=BLUE,label='class 0'); ax.scatter([],[],s=22,marker='s',color=OR,label='class 1'); ax.scatter([],[],s=60,marker='*',color=RED,label='new car')
h_,l_=ax.get_legend_handles_labels(); fig.legend(h_,l_,loc='center left',bbox_to_anchor=(0.7,0.55),frameon=False,handlelength=1.2,labelspacing=0.5,fontsize=9)
sv(fig,'f542b.png',270)

# ======== 54.7-1 algorithm flow chart (fit chain, predict chain)
fig=plt.figure(figsize=(3.5,1.55),dpi=240); ax=fig.add_axes([0,0,1,1]); ax.axis('off'); ax.set_xlim(-0.1,12.6); ax.set_ylim(-0.1,6.6)
ax.text(0.0,6.15,'fit$(\\mathbf{X},\\mathbf{y})$',fontsize=9.5,color=NAVY,weight='bold',va='center')
bw,bh=2.7,1.7; xs=[0.0,3.1,6.2,9.3]; yt,yb_=3.9,0.2
lab=['fit\n$T_1$','fit\n$T_2$','fit\n$f_m$']
for k,(xx,t) in enumerate(zip(xs[:3],lab)): box(ax,xx,yt,bw,bh,t,fs=9,fc='#E6F4EC',ec=GREEN)
for k in range(2):
    arr(ax,(xs[k]+bw+0.02,yt+bh/2),(xs[k+1]-0.02,yt+bh/2),c=NAVY)
ax.text(9.3,yt+bh/2,'stores\n$\\theta_1,\\theta_2,\\theta_m$',fontsize=9,color=GREEN,ha='left',va='center',weight='bold')
ax.text(9.3,2.95,'predict$(\\mathbf{x})$',fontsize=9.5,color=NAVY,weight='bold',va='center',ha='left')
lab2=['$T_1$\ntransform','$T_2$\ntransform','$f_m$\npredict']
for k,(xx,t) in enumerate(zip(xs[:3],lab2)): box(ax,xx,yb_,bw,bh,t,fs=9,fc=LB,ec=BLUE)
for k in range(2): arr(ax,(xs[k]+bw+0.02,yb_+bh/2),(xs[k+1]-0.02,yb_+bh/2),c=NAVY)
arr(ax,(xs[2]+bw+0.02,yb_+bh/2),(xs[3]+0.5,yb_+bh/2),c=NAVY); ax.text(xs[3]+0.6,yb_+bh/2,'$\\hat{y}$',fontsize=10,color=NAVY,va='center')
for k in range(3): ax.plot([xs[k]+bw/2]*2,[yb_+bh+0.65,yt-0.05],color=GREEN,lw=1.0,ls=':'); arr(ax,(xs[k]+bw/2,yb_+bh+0.7),(xs[k]+bw/2,yb_+bh+0.05),c=GREEN,lw=0.8)
sv(fig,'f571.png',290)

# ======== 54.10-1 how large is the leak?
fig,axs=plt.subplots(1,2,figsize=(4.7,1.7),dpi=240,sharey=True)
for ax,key,xl,tt in ((axs[0],'vs_p','features $p$ (60 rows)',None),(axs[1],'vs_n','rows $n$ (1 000 features)',None)):
    S=SW['summary'][key]; ks=sorted(S,key=float); xv=[float(k) for k in ks]
    ax.plot(xv,[S[k][0] for k in ks],color=RED,marker='o',ms=3.8,lw=1.3,label='selected first')
    ax.plot(xv,[S[k][1] for k in ks],color=GREEN,marker='s',ms=3.8,lw=1.3,ls='--',label='inside the pipeline')
    ax.axhline(0.5,color=GREY,lw=0.8,ls=':'); ax.set_xscale('log'); ax.set_xlabel(xl); xt=[float(k) for k in ks]; ax.set_xticks([xt[0],xt[2],xt[-1]] if key=='vs_p' else [xt[0],xt[2],xt[-1]]); ax.set_xticklabels([f'{int(v):,}'.replace(',',' ') for v in ([xt[0],xt[2],xt[-1]])]); ax.minorticks_off(); grid(ax); ax.set_ylim(0.4,1.0)
axs[0].set_ylabel('5-fold accuracy'); h_,l_=axs[1].get_legend_handles_labels(); fig.legend(h_,l_,loc='lower center',ncol=2,frameon=False,handlelength=1.6,fontsize=8.8,bbox_to_anchor=(0.55,0.9))
fig.tight_layout(w_pad=0.8); sv(fig,'f5102.png',420)
NUM['vs_p']=SW['summary']['vs_p']; NUM['vs_n']=SW['summary']['vs_n']

# ======== 54.11-1 the case: honest, leaky, fresh
rows=[]
for line in open('verify/case_cars.out').read().splitlines()[1:]:
    lab=line[:34].strip(); v=line[34:].split(); rows.append((lab,[None if t=='-' else float(t) for t in v]))
fig,ax=plt.subplots(figsize=(4.5,1.8),dpi=240)
yl=np.arange(len(rows))[::-1]
for (lab,(h,l,f)),yy in zip(rows,yl):
    if l is not None: ax.plot([h,l],[yy,yy],color=GRID,lw=2.0,zorder=1)
    ax.scatter(h,yy,s=30,color=GREEN,marker='s',zorder=3)
    if l is not None: ax.scatter(l,yy,s=34,color=RED,marker='^',zorder=3)
    ax.scatter(f,yy,s=34,facecolor='white',edgecolor=NAVY,marker='o',lw=1.0,zorder=3)
ax.set_yticks(yl); ax.set_yticklabels(['one-hot: steps first','target: fit, transform','target: fit_transform','one-hot: log target'],fontsize=9); ax.set_xlabel('RMSE (thousand)'); ax.set_xlim(3.1,4.15); grid(ax)
ax.scatter([],[],s=30,color=GREEN,marker='s',label='honest CV'); ax.scatter([],[],s=34,color=RED,marker='^',label='leaky CV'); ax.scatter([],[],s=34,facecolor='white',edgecolor=NAVY,marker='o',label='fresh cars')
ax.legend(loc='lower center',bbox_to_anchor=(0.5,1.0),ncol=3,handletextpad=0.1,columnspacing=1.0,fontsize=9,borderaxespad=0.1)
fig.tight_layout(); sv(fig,'f5111.png',340)
NUM['case_rows']={r[0]:r[1] for r in rows}
json.dump({'num':NUM,'font_check':CHK},open('verify/fig_numbers.json','w'),indent=1)
for k,v in CHK.items(): print(k,v)
