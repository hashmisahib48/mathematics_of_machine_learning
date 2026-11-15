# Figures of Chapter 43, second part (gallery, validation curves, case studies). Run: python3 fig/figs_b.py [part ...]  (PNGs into ch43/, numbers into verify/fig_numbers.json)
import sys,os,json,io,runpy,warnings,contextlib; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from matplotlib.patches import Ellipse, Polygon, Rectangle
from sklearn.decomposition import PCA, NMF, FastICA, FactorAnalysis, MiniBatchDictionaryLearning
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
plt.rcParams.update({'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.labelsize':9,'legend.fontsize':8.5})
ARGS=sys.argv[1:]; want=lambda k: (not ARGS) or (k in ARGS)
NUMF='verify/fig_numbers_b.json'; NUM=json.load(open(NUMF)) if (ARGS and os.path.exists(NUMF)) else {}; CHK=json.load(open('verify/fig_textsize_b.json')) if (ARGS and os.path.exists('verify/fig_textsize_b.json')) else {}
def sv(fig,name,printed_px):
    """save and record the smallest printed font size (pt)"""
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image
    with Image.open(name) as im: w=im.size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def quiet(path):
    with contextlib.redirect_stdout(io.StringIO()): return runpy.run_path(path)
DX,DY=load_digits(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(DX,DY,test_size=0.25,random_state=0,stratify=DY)
LG='#E3F4EA'

# ---------------------------------------------------------------- 43.11-1 the knob k
if want('f438'):
    KS=[4,9,16,25]; fig=plt.figure(figsize=(5.4,1.75),dpi=240); NUM['gal']={}
    gs0=fig.add_gridspec(1,4,wspace=0.12,left=0.01,right=0.99,top=0.80,bottom=0.17)
    for c_,k in enumerate(KS):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',ConvergenceWarning)
            m=NMF(k,max_iter=1000,random_state=0).fit(Xtr)
        H=m.components_; te=float(np.linalg.norm(Xte-m.transform(Xte)@H)); r=int(np.sqrt(k))
        NUM['gal'][str(k)]={'train':float(m.reconstruction_err_),'test':te,'h_zero':float((H==0).mean()),'n_iter':int(m.n_iter_)}
        gs=gs0[0,c_].subgridspec(r,r,wspace=0.05,hspace=0.05)
        for t in range(k):
            a_=fig.add_subplot(gs[t//r,t%r]); a_.imshow(H[t].reshape(8,8),cmap='gray_r',vmin=0,vmax=H[t].max(),interpolation='nearest'); a_.set_xticks([]); a_.set_yticks([])
            for sp_ in a_.spines.values(): sp_.set_color(GRID); sp_.set_linewidth(0.5)
            if t==(r-1)//2*0: pass
        x0=0.01+c_*0.245+0.1
        fig.text(0.01+(c_+0.5)*0.2475,0.93,f'k = {k}',fontsize=9,color=PUR,ha='center',va='center',weight='bold')
        fig.text(0.01+(c_+0.5)*0.2475,0.05,f'test error {te:.0f}',fontsize=8.8,color=RED,ha='center',va='center')
    sv(fig,'f438.png',470)

# ---------------------------------------------------------------- 43.11-2 choosing k and K
if want('f439'):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation as LDA
    from sklearn.metrics import adjusted_rand_score as ari
    ks=[1,2,3,4,6,8,10,12,16,20,25,30]; en=[]; ep=[]
    for k in ks:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',ConvergenceWarning)
            m=NMF(k,max_iter=1000,random_state=0).fit(Xtr)
        en.append(float(np.linalg.norm(Xte-m.transform(Xte)@m.components_)))
        p=PCA(k).fit(Xtr); ep.append(float(np.linalg.norm(Xte-p.inverse_transform(p.transform(Xte)))))
    sys.path.insert(0,'code'); from corpus import make_docs
    docs,hid=make_docs(0); X=CountVectorizer(stop_words='english',min_df=2).fit_transform(docs); pure=hid>=0
    Ks=list(range(2,9)); am=[];asd=[];pm=[]
    for K_ in Ks:
        r=[];pp=[]
        for sd in range(5):
            l=LDA(K_,max_iter=100,random_state=sd).fit(X); r.append(ari(hid[pure],l.transform(X)[pure].argmax(1))); pp.append(l.bound_)
        am.append(float(np.mean(r))); asd.append(float(np.std(r))); pm.append(float(np.mean(pp)))
    NUM['val']={'ks':ks,'nmf_test':[round(v,1) for v in en],'pca_test':[round(v,1) for v in ep],'Ks':Ks,'ari':[round(v,3) for v in am],'ari_sd':[round(v,3) for v in asd],'perp':[round(v,1) for v in pm]}
    fig,axs=plt.subplots(1,2,figsize=(5.4,2.25),dpi=240)
    ax=axs[0]; ax.plot(ks,en,'-o',color=GREEN,ms=3.2,lw=1.4,label='NMF'); ax.plot(ks,ep,'--s',color=BLUE,ms=3.0,lw=1.2,label='PCA')
    ax.set_xlabel('number of parts or components $k$'); ax.set_ylabel('test reconstruction error'); ax.set_ylim(0,None); grid(ax); ax.legend(loc='upper right',fontsize=8.5,handlelength=1.8)
    ax=axs[1]; ax.errorbar(Ks,am,yerr=asd,color=PUR,marker='o',ms=3.2,lw=1.4,capsize=2,label='agreement with hidden topics (ARI)')
    ax.set_xlabel('number of topics $K$'); ax.set_ylabel('adjusted Rand index',color=PUR); ax.set_ylim(0,1.05); grid(ax)
    ax2=ax.twinx(); ax2.plot(Ks,pm,'--^',color=RED,ms=3.0,lw=1.1); ax2.set_ylabel('perplexity (training)',color=RED); ax2.spines['right'].set_visible(True); ax2.tick_params(axis='y',colors=RED); ax2.set_ylim(90,150)
    ax.set_title('LDA on the invented corpus',fontsize=8.8,color=NAVY)
    fig.tight_layout(w_pad=1.2); sv(fig,'f439.png',470)

# ---------------------------------------------------------------- 43.12-1 digits: parts against components
if want('f4310'):
    nm=NMF(16,max_iter=1000,random_state=0).fit(Xtr); pc=PCA(16).fit(Xtr)
    fig=plt.figure(figsize=(5.4,2.75),dpi=240)
    for b_,(M,cm,ttl) in enumerate(((nm.components_,'gray_r','NMF: 16 parts (rows of $\\mathbf{H}$), all weights ≥ 0'),(pc.components_,'RdBu_r','PCA: 16 components, red positive and blue negative'))):
        gs=fig.add_gridspec(2,8,wspace=0.06,hspace=0.06,left=0.01,right=0.99,top=0.97-b_*0.5-0.04,bottom=0.53-b_*0.5)
        fig.text(0.5,0.985-b_*0.5,ttl,fontsize=8.8,color=NAVY,ha='center',va='center')
        for t in range(16):
            a_=fig.add_subplot(gs[t//8,t%8]); v=M[t].reshape(8,8)
            if b_==0: a_.imshow(v,cmap=cm,vmin=0,vmax=v.max(),interpolation='nearest')
            else: m_=np.abs(v).max(); a_.imshow(v,cmap=cm,vmin=-m_,vmax=m_,interpolation='nearest')
            a_.set_xticks([]); a_.set_yticks([])
            for sp_ in a_.spines.values(): sp_.set_color(GRID); sp_.set_linewidth(0.5)
    sv(fig,'f4310.png',470)

# ---------------------------------------------------------------- 43.12-2 two voices
if want('f4311'):
    I=quiet('code/ica_sklearn.py'); t=I['t']; S=I['S']; X=I['X']; Se=I['Se']
    Sp=PCA(2).fit_transform(X); n_=1000
    def align(E):
        c=np.corrcoef(E.T,S.T)[:2,2:]; idx=np.abs(c).argmax(0); out=np.zeros_like(S)   # estimate best matching source j
        for j in range(2): e=E[:,idx[j]]; out[:,j]=np.sign(c[idx[j],j])*(e-e.mean())/e.std()
        return out
    rows=[(S,'sources'),(X,'microphones'),(align(Sp),'PCA scores'),(align(Se),'FastICA estimates')]
    fig,axs=plt.subplots(4,2,figsize=(5.4,3.05),dpi=240,sharex=True)
    for r_,(D_,nm_) in enumerate(rows):
        for c_ in range(2):
            ax=axs[r_,c_]; ax.plot(t[:n_],D_[:n_,c_],color=BLUE if r_<2 else (OR if r_==2 else GREEN),lw=1.0)
            if r_>=2: ax.plot(t[:n_],S[:n_,c_],color=GRID,lw=0.9,zorder=0)
            ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.grid(False)
            if c_==0: ax.set_ylabel(nm_,fontsize=8.6,rotation=0,ha='right',va='center',labelpad=4)
    axs[0,0].set_title('signal 1 (sine)',fontsize=8.8,color=NAVY); axs[0,1].set_title('signal 2 (sawtooth)',fontsize=8.8,color=NAVY)
    axs[3,0].set_xlabel('time'); axs[3,1].set_xlabel('time')
    fig.tight_layout(h_pad=0.3,w_pad=0.5); sv(fig,'f4311.png',470)

# ---------------------------------------------------------------- 43.12-3 topics
if want('f4312'):
    L=quiet('code/lda_sklearn.py'); lda=L['lda']; words=L['words']; th=L['th']; hid=L['hidden']; docs=L['docs']
    import collections
    names=['cooking','football','astronomy','finance']; mp={}
    for k in range(4): mp[k]=names[collections.Counter(hid[(hid>=0)&(th.argmax(1)==k)]).most_common(1)[0][0]]
    cols=[BLUE,OR,GREEN,PUR]; Bt=lda.components_/lda.components_.sum(1,keepdims=True)
    NUM['topics']={'map':{str(k):mp[k] for k in range(4)},'top':{str(k):[(words[j],round(float(Bt[k,j]),3)) for j in np.argsort(-Bt[k])[:6]] for k in range(4)}}
    fig=plt.figure(figsize=(5.4,2.7),dpi=240)
    gl=fig.add_gridspec(2,2,left=0.095,right=0.60,top=0.95,bottom=0.09,wspace=0.85,hspace=0.50)
    for k in range(4):
        ax=fig.add_subplot(gl[k//2,k%2]); top=np.argsort(-Bt[k])[:6][::-1]; ax.barh(range(6),Bt[k,top],color=cols[k],height=0.7); ax.set_yticks(range(6)); ax.set_yticklabels(words[top],fontsize=8.5); ax.set_xticks([])
        ax.set_title(f'topic {k}: {mp[k]}',fontsize=8.8,color=cols[k],pad=2,weight='bold'); ax.spines['bottom'].set_visible(False)
    pure_idx=[]; 
    for t_ in range(4): pure_idx+= [j for j in np.flatnonzero(hid==[n for n in range(4) if names[n]==mp[t_]][0])][:2]
    mixed=list(np.flatnonzero(hid<0)[:4]); sel=pure_idx+mixed
    ax=fig.add_axes([0.77,0.09,0.22,0.86]); left=np.zeros(len(sel))
    for k in range(4): ax.barh(range(len(sel)),th[sel,k],left=left,color=cols[k],height=0.75,edgecolor='white',lw=0.4); left+=th[sel,k]
    ax.set_yticks(range(len(sel))); ax.set_yticklabels([(names[hid[j]] if hid[j]>=0 else 'mixed') for j in sel],fontsize=8.5); ax.set_xlim(0,1); ax.set_xticks([0,0.5,1]); ax.invert_yaxis(); ax.set_title('topic mix of 12 documents',fontsize=8.8,color=NAVY,pad=3)
    sv(fig,'f4312.png',470)

if ARGS or True:
    pass
json.dump(NUM,open(NUMF,'w'),indent=1); json.dump(CHK,open('verify/fig_textsize_b.json','w'),indent=1)
print('figs ok',[k for k,v in CHK.items() if v['min_font_printed']<7.5])
