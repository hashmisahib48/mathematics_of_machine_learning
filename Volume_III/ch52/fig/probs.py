# Problem figures of Chapter 52. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.utils import murmurhash3_32
from corpus import messages
plt.rcParams.update({'xtick.labelsize':9,'ytick.labelsize':9,'axes.labelsize':9.5,'legend.fontsize':9})
A={}; CHK={}
def small(w=2.4,h=1.15): return plt.subplots(figsize=(w,h),dpi=240)
def sv(fig,name):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()); fig.tight_layout(pad=0.4); save(fig,name)
    from PIL import Image
    with Image.open(name) as im: w,h=im.size
    pw=min(1.55*135,235*w/h); CHK[name]=round(fs*pw/96/(w/240),1)
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
r4=lambda v:[round(float(x),4) for x in np.ravel(v)]
msgs,y=messages(); y=np.array(y); n=len(msgs)

# 52A.1 tf-idf with the other options
docs=["tea tea milk","milk sugar","tea sugar sugar sugar","tea"]
Xc=CountVectorizer().fit_transform(docs)
out={}
for nm,kw in (('default',{}),('plain',dict(smooth_idf=False,sublinear_tf=True))):
    t=TfidfTransformer(**kw).fit(Xc); out[nm]={'idf':r4(t.idf_),'row1':r4(t.transform(Xc)[0].toarray())}
A['52A.1']={'terms':['milk','sugar','tea'],'counts':Xc.toarray().tolist(),'df':[2,2,3],**out}
fig,ax=small(2.4,1.2); w=0.36; x=np.arange(3)
ax.bar(x-w/2,out['default']['row1'],w,color=BLUE,label='default'); ax.bar(x+w/2,out['plain']['row1'],w,color=GREEN,label='sublinear, plain idf')
ax.set_xticks(x); ax.set_xticklabels(['milk','sugar','tea']); ax.set_ylabel('row 1 of $\\mathbf{Z}$'); ax.set_ylim(0,1.75); ax.legend(loc='upper left',ncol=1,fontsize=8.5,handlelength=1.0,borderaxespad=0.1); grid(ax); sv(fig,'p_52A1.png')

# 52A.2 expected number of occupied buckets
cv=CountVectorizer().fit(msgs); words=cv.get_feature_names_out(); V=len(words)
hs=np.array([murmurhash3_32(w_,seed=0,positive=False) for w_ in words]).astype(np.int64)
res={}
for m in (64,1024):
    exp=m*(1-(1-1/m)**V); occ=len(set(np.abs(hs)%m)); res[str(m)]={'expected_occupied':round(float(exp),2),'observed_occupied':occ,'expected_lost':round(float(V-exp),2),'observed_lost':V-occ}
A['52A.2']={'V':V,**res}
fig,ax=small(2.4,1.15); x=np.arange(2); w=0.36
ax.bar(x-w/2,[res[k]['expected_occupied'] for k in ('64','1024')],w,color=PUR,label='Eq. 52.5-3'); ax.bar(x+w/2,[res[k]['observed_occupied'] for k in ('64','1024')],w,color=BLUE,label='observed')
ax.set_ylim(0,330); ax.set_xticks(x); ax.set_xticklabels(['$m=64$','$m=1024$']); ax.set_ylabel('occupied'); ax.legend(loc='upper left',fontsize=8.5,handlelength=1.0,borderaxespad=0.1); grid(ax); sv(fig,'p_52A2.png')

# 52A.3 counting n-grams
text="stir the soup and taste it"; rows={}
for stop in (None,'english'):
    for r in ((1,1),(1,2),(1,3)):
        rows['%s %d-%d'%('stop' if stop else 'all',*r)]=len(CountVectorizer(stop_words=stop,ngram_range=r).build_analyzer()(text))
A['52A.3']={'text':text,'counts':rows,'tokens_all':CountVectorizer().build_analyzer()(text),'tokens_stop':CountVectorizer(stop_words='english').build_analyzer()(text)}
fig,ax=small(2.4,1.15); x=np.arange(3); w=0.36
ax.bar(x-w/2,[rows['all 1-%d'%k] for k in (1,2,3)],w,color=BLUE,label='all tokens'); ax.bar(x+w/2,[rows['stop 1-%d'%k] for k in (1,2,3)],w,color=GREEN,label='stop words out')
ax.set_xticks(x); ax.set_xticklabels(['(1,1)','(1,2)','(1,3)']); ax.set_ylabel('features'); ax.set_ylim(0,27); ax.legend(loc='upper left',fontsize=8.5,handlelength=1.0,borderaxespad=0.1); grid(ax); sv(fig,'p_52A3.png')

# 52B.1 sparse against dense
mem={}
for r in ((1,1),(1,2)):
    X=CountVectorizer(ngram_range=r).fit_transform(msgs); mem['%d-%d'%r]={'shape':list(X.shape),'nnz':int(X.nnz),'csr_bytes':int(X.data.nbytes+X.indices.nbytes+X.indptr.nbytes),'dense_bytes':int(X.shape[0]*X.shape[1]*X.dtype.itemsize),'dtype':str(X.dtype)}
    mem['%d-%d'%r]['ratio']=round(mem['%d-%d'%r]['dense_bytes']/mem['%d-%d'%r]['csr_bytes'],2); mem['%d-%d'%r]['share']=round(X.nnz/np.prod(X.shape),4)
A['52B.1']=mem
fig,ax=small(2.4,1.15); x=np.arange(2); w=0.36
ax.bar(x-w/2,[mem[k]['dense_bytes']/1e3 for k in ('1-1','1-2')],w,color=RED,label='dense'); ax.bar(x+w/2,[mem[k]['csr_bytes']/1e3 for k in ('1-1','1-2')],w,color=GREEN,label='sparse (CSR)')
ax.set_yscale('log'); ax.set_ylim(1,20000); ax.set_xticks(x); ax.set_xticklabels(['(1,1)','(1,2)']); ax.set_ylabel('kilobytes'); ax.legend(loc='upper left',fontsize=8.5,handlelength=1.0,borderaxespad=0.1,ncol=2,columnspacing=0.8); grid(ax); sv(fig,'p_52B1.png')

# 52B.2 streaming with the hashing trick
Xtr,Xte,ytr,yte=train_test_split(msgs,y,test_size=12,stratify=y,random_state=0)
perm=np.random.RandomState(0).permutation(len(Xtr)); Xtr=[Xtr[i] for i in perm]; ytr=ytr[perm]
hv=HashingVectorizer(n_features=2**12,alternate_sign=False,norm=None)
nb=MultinomialNB(); accs=[]; sizes=[]
for b in range(4):
    sl=slice(9*b,9*b+9); nb.partial_fit(hv.transform(Xtr[sl]),ytr[sl],classes=[0,1,2,3]); accs.append(float(nb.score(hv.transform(Xte),yte))); sizes.append(9*(b+1))
bat=make_pipeline(CountVectorizer(),MultinomialNB()).fit(Xtr,ytr).score(Xte,yte)
bat_h=MultinomialNB().fit(hv.transform(Xtr),ytr).score(hv.transform(Xte),yte)
A['52B.2']={'accuracy_after_batches':r4(accs),'sizes':sizes,'batch_vocabulary':round(float(bat),4),'batch_hashed':round(float(bat_h),4),'train':len(Xtr),'test':len(Xte)}
fig,ax=small(2.4,1.15); ax.plot(sizes,accs,'-o',color=BLUE,ms=3.8,lw=1.4); ax.axhline(bat_h,color=GREEN,lw=1.1,ls='--'); ax.set_xticks(sizes); ax.set_ylim(0.3,1.08); ax.set_xlabel('messages seen'); ax.set_ylabel('test accuracy'); grid(ax); sv(fig,'p_52B2.png')

# 52C.1 unseen words
rng=np.random.RandomState(0); sizes=[6,12,18,24,30,36]; un=[]
for s_ in sizes:
    f=[]
    for rep in range(40):
        idx=rng.permutation(n); tr=idx[:s_]; te=idx[s_:]
        vc=CountVectorizer().fit([msgs[i] for i in tr]); Xt=CountVectorizer().build_analyzer()
        toks=[w_ for i in te for w_ in Xt(msgs[i])]; f.append(np.mean([w_ not in vc.vocabulary_ for w_ in toks]))
    un.append(float(np.mean(f)))
A['52C.1']={'train_sizes':sizes,'unseen_token_share':r4(un)}
fig,ax=small(2.4,1.15); ax.plot(sizes,un,'-o',color=RED,ms=3.8,lw=1.4); ax.set_xlabel('training messages'); ax.set_ylabel('unseen tokens'); ax.set_ylim(0,0.8); grid(ax); sv(fig,'p_52C1.png')

# 52C.2 stop words and counts with naive Bayes
cvr=RepeatedStratifiedKFold(n_splits=4,n_repeats=5,random_state=0); sc={}
for nm,vec in (('counts',CountVectorizer()),('counts, stop words out',CountVectorizer(stop_words='english'))):
    sc[nm]=cross_val_score(make_pipeline(vec,MultinomialNB()),msgs,y,cv=cvr)
for nm,vec in (('tf-idf',TfidfVectorizer()),('tf-idf, stop words out',TfidfVectorizer(stop_words='english'))):
    sc[nm]=cross_val_score(make_pipeline(vec,MultinomialNB()),msgs,y,cv=cvr)
A['52C.2']={k:{'mean':round(float(v.mean()),4),'sd':round(float(v.std()),4)} for k,v in sc.items()}
fig,ax=small(2.4,1.2)
for k,(nm,v) in enumerate(sc.items()):
    ax.plot(k+np.linspace(-0.2,0.2,len(v)),v,'o',color=BLUE,ms=2.4,alpha=0.7); ax.plot([k-0.3,k+0.3],[v.mean()]*2,color=RED,lw=1.6)
ax.set_xticks(range(4)); ax.set_xticklabels(['cnt','cnt\nstop','tf-idf','tf-idf\nstop'],fontsize=8.5); ax.set_ylabel('fold accuracy'); ax.set_ylim(0.6,1.04); ax.set_xlim(-0.6,3.6); grid(ax); sv(fig,'p_52C2.png')

json.dump(A,open('verify/problem_answers.json','w'),indent=1); json.dump(CHK,open('verify/probfig_textsize.json','w'),indent=1)
print({k:v for k,v in CHK.items() if v<7.5}); print(json.dumps(A))
