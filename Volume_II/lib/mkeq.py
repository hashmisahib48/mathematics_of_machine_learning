r"""Equation images. Usage:
   import sys; sys.path.insert(0,'<root>/lib'); from mkeq import mkeq
   mkeq({'e621': r'$E(f)=E_0\,(1-f)$', 'e622': [r'$line 1$', r'$line 2$']}, outdir)
Multi-line equations are given as a list. Keep each image narrower than ~5.2 in at fontsize 11.5
(the builder scales any wider image down to 400 px)."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, os
plt.rcParams['mathtext.fontset']='stix'; plt.rcParams['font.family']='STIXGeneral'
def mkeq(E,outdir='.'):
    for k,v in E.items():
        lst=isinstance(v,list); n=len(v) if lst else 1
        fig=plt.figure(figsize=(0.01,0.01 if not lst else 0.30*n))
        if lst:
            for j,t in enumerate(v): fig.text(0,0.9-(0.62 if n==2 else 1.15/n)*j,t,fontsize=11.5)
        else: fig.text(0,0,v,fontsize=11.5)
        fig.savefig(os.path.join(outdir,k+'.png'),dpi=300,bbox_inches='tight',pad_inches=0.02,facecolor='white'); plt.close(fig)

def matimg(name, items, outdir='.', fs=11.5, rh=0.2):
    """Equation containing bracketed matrices (mathtext has no pmatrix).
    items: list of strings (mathtext, e.g. r'$\\mathbf{X}=$') and matrices (list of rows of strings, mathtext without $).
    Example: matimg('m951', [r'$\\mathbf{X}=$', [['1','1'],['1','2']], r'$,\\quad \\mathbf{y}=$', [['3'],['4']]])"""
    from matplotlib.lines import Line2D
    fig=plt.figure(figsize=(9,2.2),dpi=300); r=fig.canvas.get_renderer(); Wf,Hf=9.0,2.2; x=0.1; yc=1.1
    def wid(t): return t.get_window_extent(renderer=r).width/300
    for it in items:
        if isinstance(it,str):
            t=fig.text(x/Wf,yc/Hf,it,fontsize=fs,va='center'); x+=wid(t)+0.05
        else:
            nr=len(it); nc=len(it[0]); x0=x; x+=0.09; top=yc+rh*(nr-1)/2
            for j in range(nc):
                ts=[fig.text(0,0,'$'+it[i][j]+'$',fontsize=fs,va='center',ha='center') for i in range(nr)]; cw=max(wid(t) for t in ts)
                for i,t in enumerate(ts): t.set_position(((x+cw/2)/Wf,(top-i*rh)/Hf))
                x+=cw+0.16
            x-=0.16-0.09; yb,yt=top-(nr-1)*rh-0.13,top+0.13
            for xa,d in ((x0,1),(x,-1)):
                fig.add_artist(Line2D([(xa+0.05*d)/Wf,xa/Wf,xa/Wf,(xa+0.05*d)/Wf],[yt/Hf,yt/Hf,yb/Hf,yb/Hf],color='black',lw=0.8))
            x+=0.07
    fig.savefig(os.path.join(outdir,name+'.png'),dpi=300,bbox_inches='tight',pad_inches=0.03,facecolor='white'); plt.close(fig)
