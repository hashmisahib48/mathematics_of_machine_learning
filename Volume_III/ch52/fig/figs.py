# All figures of Chapter 52. Run: python3 fig/figs.py  (PNGs into ch52/, numbers into verify/fig_numbers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.utils import murmurhash3_32
from corpus import messages, make_docs, TOPIC_NAMES
plt.rcParams.update({'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.labelsize':9,'legend.fontsize':8.5})
NUM={}; CHK={}
PX={'f520.png':360,'f521.png':340,'f522.png':380,'f523.png':400,'f524.png':420,'f525.png':410,'f526.png':390}
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
TC=[BLUE,OR,GREEN,RED]; TM=['o','s','^','D']
msgs,y=messages(); y=np.array(y); n=len(msgs)
cvz=CountVectorizer(); X=cvz.fit_transform(msgs); words=cvz.get_feature_names_out(); V=len(words)
df=np.asarray((X>0).sum(0)).ravel()

# ======== 52.0-1 opener: the matrix of 48 messages x 216 words, and how rare the words are
fig,axs=plt.subplots(1,2,figsize=(4.7,2.15),dpi=240,gridspec_kw={'width_ratios':[1.35,1]})
ax=axs[0]; Xd=(X.toarray()>0).astype(float)
# order words by the topic that uses them most, then by document frequency
tc=np.array([[Xd[y==k,j].sum() for k in range(4)] for j in range(V)]); main=tc.argmax(1)
order=np.lexsort((-df,main)); ax.imshow(Xd[:,order],aspect='auto',cmap=LinearSegmentedColormap.from_list('b',['#FFFFFF',NAVY]),interpolation='nearest')
for k in (1,2,3): ax.axhline(12*k-0.5,color=RED,lw=0.7)
ax.set_xticks([]); ax.set_yticks([5.5,17.5,29.5,41.5]); ax.set_yticklabels(TOPIC_NAMES,fontsize=8.5); ax.tick_params(axis='y',length=0)
ax.set_xlabel('%d words'%V); ax.set_title('%d messages, %.1f %% non-zero'%(n,100*X.nnz/(n*V)),fontsize=9,color=NAVY)
for s in ax.spines.values(): s.set_visible(False)
ax=axs[1]; bins=[(1,1),(2,2),(3,4),(5,9),(10,48)]; lab=['1','2','3-4','5-9','10+']
cnt=[int(((df>=a)&(df<=b)).sum()) for a,b in bins]
ax.bar(range(5),cnt,color=BLUE,width=0.7); ax.set_xticks(range(5)); ax.set_xticklabels(lab)
for i,c_ in enumerate(cnt): ax.text(i,c_+3,str(c_),ha='center',fontsize=8.5,color=NAVY)
ax.set_ylim(0,160); ax.set_xlabel('messages containing the word'); ax.set_ylabel('number of words'); ax.set_title('most words are rare',fontsize=9,color=NAVY); grid(ax)
NUM['opener']={'bins':lab,'counts':cnt,'share':float(X.nnz/(n*V))}
fig.tight_layout(w_pad=1.0); sv(fig,'f520.png',420)

# ======== 52.2-1 intuition: length against direction
d1=np.array([2.,1.]); d2=3*d1; d3=np.array([1.,2.]); u=lambda v:v/np.linalg.norm(v)
NUM['geom']={'euclid_d1_d2':float(np.linalg.norm(d1-d2)),'euclid_d1_d3':float(np.linalg.norm(d1-d3)),'cos_d1_d2':float(u(d1)@u(d2)),'cos_d1_d3':float(u(d1)@u(d3)),'unit_d1':u(d1).round(3).tolist(),'unit_d3':u(d3).round(3).tolist()}
fig,axs=plt.subplots(1,2,figsize=(4.4,1.95),dpi=240)
ax=axs[0]; ax.set_aspect('equal'); ax.set_xlim(-0.3,6.9); ax.set_ylim(-0.3,3.7)
for v,col,nm,off in ((d1,BLUE,'A',(0.12,-0.45)),(d2,BLUE,'B',(0.1,-0.5)),(d3,OR,'C',(-0.45,0.05))):
    arr(ax,(0,0),tuple(v),c=col,lw=1.3,ms=7); ax.text(v[0]+off[0],v[1]+off[1],nm,fontsize=9.5,color=col,weight='bold')
ax.plot([d1[0],d2[0]],[d1[1],d2[1]],color=RED,lw=1.0,ls='--'); ax.plot([d1[0],d3[0]],[d1[1],d3[1]],color=RED,lw=1.0,ls='--')
ax.text(3.3,0.9,'|AB| = %.2f'%NUM['geom']['euclid_d1_d2'],fontsize=8.5,color=RED,ha='left'); ax.text(1.25,2.75,'|AC| = %.2f'%NUM['geom']['euclid_d1_d3'],fontsize=8.5,color=RED,ha='left')
ax.set_xlabel('count of "goal"'); ax.set_ylabel('count of "bank"'); ax.set_title('counts',fontsize=9,color=NAVY); ax.set_xticks([0,2,4,6]); ax.set_yticks([0,1,2,3]); grid(ax)
ax=axs[1]; ax.set_aspect('equal'); th=np.linspace(0,np.pi/2,100); ax.plot(np.cos(th),np.sin(th),color=GRID,lw=1.2)
for v,col,nm,off in ((d1,BLUE,'A, B',(0.05,-0.14)),(d3,OR,'C',(-0.12,0.05))):
    p=u(v); arr(ax,(0,0),tuple(p),c=col,lw=1.3,ms=7); ax.text(p[0]+off[0],p[1]+off[1],nm,fontsize=9.5,color=col,weight='bold')
ax.set_xlim(-0.1,1.35); ax.set_ylim(-0.1,1.25); ax.text(0.5,1.08,'cos(A, C) = %.2f'%NUM['geom']['cos_d1_d3'],fontsize=8.5,color=GREEN,ha='left')
ax.set_xlabel('"goal" share'); ax.set_ylabel('"bank" share'); ax.set_title('unit length',fontsize=9,color=NAVY); ax.set_xticks([0,0.5,1]); ax.set_yticks([0,0.5,1]); grid(ax)
fig.tight_layout(w_pad=1.0); sv(fig,'f521.png',400)

# ======== 52.4-1 idf curve and the weights of one message
tv=TfidfVectorizer(); Z=tv.fit_transform(msgs); idf=tv.idf_
fig,axs=plt.subplots(1,2,figsize=(5.0,2.0),dpi=240,gridspec_kw={'width_ratios':[1,1.15]})
ax=axs[0]; dd=np.arange(1,n+1); ax.plot(dd,np.log((1+n)/(1+dd))+1,color=GREEN,lw=1.5,label='smooth (default)'); ax.plot(dd,np.log(n/dd)+1,color=GREY,lw=1.2,ls='--',label='`smooth_idf=False`'.replace('`',''))
mw=np.nonzero(X[0].toarray().ravel())[0]; dfm={str(words[j]):int(df[j]) for j in mw}
for wd,off in (('the',(-9,0.55)),('pan',(1.5,0.3)),('sauce',(1.5,0.3))):
    d_=dfm[wd]; ax.plot(d_,np.log((1+n)/(1+d_))+1,'o',color=RED,ms=4.5,zorder=5); ax.text(d_+off[0],np.log((1+n)/(1+d_))+1+off[1],wd,fontsize=8.5,color=RED)
NUM['idf_words']={wd:{'df':dfm[wd],'idf':round(float(np.log((1+n)/(1+dfm[wd]))+1),4)} for wd in dfm}
ax.set_xlabel('document frequency df'); ax.set_ylabel('idf'); ax.set_ylim(0.7,5.3); ax.legend(loc='upper right',handlelength=1.5,fontsize=8,borderaxespad=0.2); grid(ax)
ax=axs[1]; cnt=X[0].toarray().ravel().astype(float); cu=cnt/np.linalg.norm(cnt); zr=Z[0].toarray().ravel()
ws=[j for j in np.nonzero(cnt)[0]]; ws=sorted(ws,key=lambda j:-cu[j]); pos=np.arange(len(ws))
ax.barh(pos-0.19,[cu[j] for j in ws],height=0.36,color=BLUE,label='counts, unit length'); ax.barh(pos+0.19,[zr[tv.vocabulary_[words[j]]] for j in ws],height=0.36,color=GREEN,label='tf-idf, unit length')
ax.set_yticks(pos); ax.set_yticklabels([words[j] for j in ws]); ax.invert_yaxis(); ax.set_xlabel('entry of the row'); ax.set_xlim(0,1.15); ax.legend(loc='lower right',handlelength=1.0,fontsize=8,borderaxespad=0.2); grid(ax)
NUM['message0']={'text':msgs[0],'count_unit':{str(words[j]):round(float(cu[j]),3) for j in ws},'tfidf':{str(words[j]):round(float(zr[tv.vocabulary_[words[j]]]),3) for j in ws}}
fig.tight_layout(w_pad=0.9); sv(fig,'f522.png',430)

# ======== 52.5-1 hashing: collisions and the error of the inner product
iu=np.triu_indices(n,1); G=(normalize(X)@normalize(X).T).toarray()[iu]
ps=np.arange(3,13); H={'m':(2**ps).tolist(),'rmse_signed':[],'rmse_unsigned':[],'bias_signed':[],'bias_unsigned':[],'lost':[],'expected_lost':[],'occupied':[]}
hs=np.array([murmurhash3_32(w,seed=0,positive=False) for w in words]).astype(np.int64)
for p in ps:
    m=2**int(p); occ=len(set(np.abs(hs)%m)); H['occupied'].append(occ); H['lost'].append(V-occ); H['expected_lost'].append(V-m*(1-(1-1/m)**V))
    for alt,key in ((True,'signed'),(False,'unsigned')):
        Xh=HashingVectorizer(n_features=m,alternate_sign=alt,norm='l2').transform(msgs); d=((Xh@Xh.T).toarray()[iu]-G)
        H['rmse_'+key].append(float(np.sqrt((d**2).mean()))); H['bias_'+key].append(float(d.mean()))
NUM['hashing']=H
fig,axs=plt.subplots(1,2,figsize=(5.0,1.95),dpi=240)
ax=axs[0]; ax.plot(ps,H['bias_unsigned'],'-s',color=RED,ms=3.2,lw=1.2,label='unsigned'); ax.plot(ps,H['bias_signed'],'-o',color=GREEN,ms=3.2,lw=1.4,label='signed (default)')
ax.axhline(0,color=GREY,lw=0.8); ax.set_xlabel('$\\log_2 m$ (number of buckets)'); ax.set_ylabel('mean error of the cosine'); ax.set_title('inner products of the messages',fontsize=9,color=NAVY); ax.legend(loc='upper right',handlelength=1.4,fontsize=8,borderaxespad=0.2); grid(ax)
ax=axs[1]; ax.plot(ps,H['expected_lost'],color=PUR,lw=1.3,label='Eq. 52.5-3'); ax.plot(ps,H['lost'],'o',color=BLUE,ms=3.8,label='observed')
ax.set_yscale('log'); ax.set_xlabel('$\\log_2 m$ (number of buckets)'); ax.set_title('collisions of %d words'%V,fontsize=9,color=NAVY); ax.legend(loc='upper right',handlelength=1.4,fontsize=8,borderaxespad=0.2); grid(ax)
ax.set_ylabel('V minus occupied buckets',fontsize=8.5)
fig.tight_layout(w_pad=1.0); sv(fig,'f523.png',430)

# ======== 52.7-1 algorithm
fig,ax=plt.subplots(figsize=(5.0,1.9),dpi=240); ax.set_xlim(-0.1,10.25); ax.set_ylim(0,4.7); ax.axis('off')
bw=1.82; xs=[0.0,2.07,4.14,6.21,8.28]
top=['training\nmessages','tokens and\nn-grams','vocabulary\nand counts','weights\ntf $\\times$ idf','rows of unit\nlength $\\mathbf{Z}$']
fc=[LB,LGREY,LGREY,LGREY,'#E3F4EA']; ec=[BLUE,GREY,GREY,GREY,GREEN]
for x,t,f,e in zip(xs,top,fc,ec): box(ax,x,3.3,bw,1.3,t,fc=f,ec=e,fs=8.5)
bot=['later\nmessages','tokens and\nn-grams','stored\nvocabulary','stored idf,\nsame weights','rows of unit\nlength']
fc=['#F1F2F4',LGREY,LGREY,LGREY,'#E3F4EA']; ec=[GREY,GREY,GREY,GREY,GREEN]
for x,t,f,e in zip(xs,bot,fc,ec): box(ax,x,0.1,bw,1.3,t,fc=f,ec=e,fs=8.5)
for x in xs[:-1]:
    arr(ax,(x+bw,3.95),(x+bw+0.25,3.95),c=NAVY,lw=1,ms=7); arr(ax,(x+bw,0.75),(x+bw+0.25,0.75),c=NAVY,lw=1,ms=7)
arr(ax,(5.05,3.3),(5.05,1.4),c=PUR,lw=1.3,ms=8); ax.text(5.3,2.35,'kept in the fitted object:\nwords, $\\mathrm{df}_j$, $\\mathrm{idf}_j$',fontsize=8.5,color=PUR,va='center')
sv(fig,'f524.png',420)

# ======== 52.10-1 gallery: three knobs and the accuracy of tf-idf + logistic regression
cvr=RepeatedStratifiedKFold(n_splits=4,n_repeats=5,random_state=0)
LR=lambda: LogisticRegression(C=10,max_iter=1000)
def acc(p): s_=cross_val_score(p,msgs,y,cv=cvr); return float(s_.mean()),float(s_.std())
G={'min_df':{},'ngram':{},'hash':{}}
for k in range(1,7):
    vec=TfidfVectorizer(min_df=k); G['min_df'][k]=(*acc(make_pipeline(vec,LR())),len(vec.fit(msgs).vocabulary_))
for r in ((1,1),(1,2),(1,3),(2,2)):
    vec=TfidfVectorizer(ngram_range=r); G['ngram']['%d-%d'%r]=(*acc(make_pipeline(vec,LR())),len(vec.fit(msgs).vocabulary_))
for p_ in range(4,13):
    G['hash'][p_]=acc(make_pipeline(HashingVectorizer(n_features=2**p_,alternate_sign=False,norm=None),TfidfTransformer(),LR()))
base=G['min_df'][1][0]; NUM['gallery']=G
fig,axs=plt.subplots(1,3,figsize=(5.0,2.05),dpi=240)
ax=axs[0]; ks=list(G['min_df']); ax.plot(ks,[G['min_df'][k][0] for k in ks],'-o',color=PUR,ms=3.6,lw=1.4)
ax.set_xticks(ks); ax.set_xticklabels(['%d\n%d'%(k,G['min_df'][k][2]) for k in ks],fontsize=8); ax.set_xlabel('min_df (top)\nwords kept (bottom)',fontsize=8.5); ax.set_ylabel('4-fold accuracy'); ax.set_ylim(0.3,1.02); ax.set_title('rare words',fontsize=9,color=NAVY); grid(ax)
ax=axs[1]; ng=list(G['ngram']); ax.plot(range(len(ng)),[G['ngram'][k][0] for k in ng],'-o',color=PUR,ms=3.6,lw=1.4)
ax.set_xticks(range(len(ng))); ax.set_xticklabels(['(%s)\n%d'%(k.replace('-',','),G['ngram'][k][2]) for k in ng],fontsize=8); ax.set_xlabel('ngram_range (top)\nwords kept (bottom)',fontsize=8.5); ax.set_ylim(0.3,1.02); ax.set_yticklabels([]); ax.set_title('word pairs',fontsize=9,color=NAVY); grid(ax); ax.set_xlim(-0.4,3.4)
ax=axs[2]; hk=[4,6,8,10,12]; ax.plot(range(len(hk)),[G['hash'][k][0] for k in hk],'-o',color=PUR,ms=3.6,lw=1.4); ax.axhline(base,color=GREEN,lw=1.2,ls='--'); ax.text(4.0,0.62,'dashed: all\nwords',fontsize=8,color=GREEN,ha='right',va='center')
ax.set_xticks(range(len(hk))); ax.set_xticklabels(['$2^{%d}$'%k for k in hk],fontsize=8.5); ax.set_xlabel('n_features $m$',fontsize=8.5); ax.set_ylim(0.3,1.02); ax.set_yticklabels([]); ax.set_title('hashed columns',fontsize=9,color=NAVY); grid(ax)
fig.tight_layout(w_pad=0.6); sv(fig,'f525.png',470)

# ======== 52.11-1 case: similarity of the 60 documents and their LSA picture
docs,topic=make_docs(0); order=np.argsort(np.where(topic<0,9,topic),kind='stable'); topo=topic[order]
tv60=TfidfVectorizer(stop_words='english'); X60=tv60.fit_transform(docs); S=(X60@X60.T).toarray()
svd=TruncatedSVD(2,random_state=0); Z2=svd.fit_transform(X60)
fig,axs=plt.subplots(1,2,figsize=(5.0,2.3),dpi=240,gridspec_kw={'width_ratios':[1,1.12]})
ax=axs[0]; Sd=S[np.ix_(order,order)]; im=ax.imshow(Sd,cmap=LinearSegmentedColormap.from_list('g',['#FFFFFF',GREEN]),vmin=0,vmax=1,interpolation='nearest')
for b_ in (11.5,23.5,35.5,47.5): ax.axhline(b_,color=RED,lw=0.6); ax.axvline(b_,color=RED,lw=0.6)
ax.set_xticks([5.5,17.5,29.5,41.5,53.5]); ax.set_xticklabels(['cook','foot','astro','fin','mixed'],fontsize=8,rotation=0); ax.set_yticks([]); ax.set_title('cosine of the 60 documents',fontsize=9,color=NAVY); ax.tick_params(axis='x',length=0)
for s_ in ax.spines.values(): s_.set_visible(False)
ax=axs[1]
for k in range(4): ax.scatter(Z2[topic==k,0],Z2[topic==k,1],s=14,color=TC[k],marker=TM[k],lw=0,label=TOPIC_NAMES[k])
ax.scatter(Z2[topic<0,0],Z2[topic<0,1],s=18,facecolor='none',edgecolor=GREY,marker='o',lw=0.9,label='mixed')
ax.set_xlabel('LSA component 1'); ax.set_ylabel('LSA component 2'); ax.set_title('two latent dimensions',fontsize=9,color=NAVY); ax.legend(loc='upper left',bbox_to_anchor=(1.0,1.0),handletextpad=0.1,borderaxespad=0.0,fontsize=8,labelspacing=0.25); grid(ax)
NUM['lsa']={'ratio':svd.explained_variance_ratio_.round(4).tolist()}
fig.tight_layout(w_pad=0.8); sv(fig,'f526.png',470)

json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); json.dump(CHK,open('verify/fig_textsize.json','w'),indent=1)
print({k:v['min_font_printed'] for k,v in CHK.items()})
