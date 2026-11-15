# Facts quoted in the prose of Chapter 7 (writes verify/numbers.json). Run: python3 fig/checks.py
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE+'/..'); warnings.simplefilter('ignore')
import numpy as np
from sklearn import datasets as ds
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LinearRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
N={}
N['sizes']={f.__name__:list(f().data.shape) for f in (ds.load_iris,ds.load_diabetes,ds.load_breast_cancer,ds.load_wine,ds.load_digits)}
N['class_counts']={'iris':np.bincount(ds.load_iris().target).tolist(),'breast_cancer':np.bincount(ds.load_breast_cancer().target).tolist(),'wine':np.bincount(ds.load_wine().target).tolist(),'digits':np.bincount(ds.load_digits().target).tolist()}
d=ds.load_diabetes(); N['diabetes_target']=[float(d.target.min()),float(d.target.max())]; N['diabetes_feature_names']=d.feature_names
g=ds.load_digits(); N['digits']={'images':list(g.images.shape),'pixel_min_max':[float(g.data.min()),float(g.data.max())]}
N['breast_cancer_target_names']=ds.load_breast_cancer().target_names.tolist()
N['linnerud']={'X':list(ds.load_linnerud().data.shape),'y':list(ds.load_linnerud().target.shape)}
X,y=ds.load_iris(return_X_y=True)
Xb=X[y<2]; yb=y[y<2]
lb=LogisticRegression(max_iter=500).fit(Xb,yb); N['binary']={'decision_function':list(lb.decision_function(Xb[:4]).shape),'predict_proba':list(lb.predict_proba(Xb[:4]).shape)}
N['has']={n:{m:hasattr(e,m) for m in ('predict_proba','decision_function','transform','predict')} for n,e in [('LogisticRegression',LogisticRegression()),('RidgeClassifier',RidgeClassifier()),('SVC',SVC()),('SVC(probability=True)',SVC(probability=True)),('LinearSVC',LinearSVC()),('KMeans',KMeans(n_init=1)),('LinearRegression',LinearRegression()),('StandardScaler',StandardScaler())]}
N['pipeline_has']={m:hasattr(make_pipeline(StandardScaler(),LogisticRegression()),m) for m in ('predict','predict_proba','decision_function','transform','score')}
N['pipeline_no_clf_has_transform']=hasattr(make_pipeline(StandardScaler(),StandardScaler()),'transform')
N['column_y']={'LinearRegression_coef_shape':list(LinearRegression().fit(X[:20,:1],y[:20].reshape(-1,1)).coef_.shape),'one_over':{'38':round(1/38,4),'112':round(1/112,4)}}
with warnings.catch_warnings(record=True) as w_:
    warnings.simplefilter('always'); LogisticRegression().fit(np.array([[1.,0],[2,1],[3,0]]),[[0],[1],[0]])
N['column_y']['classifier_warning']=w_[0].category.__name__
d_=ds.load_iris(); N['bunch']={'type':type(d_).__name__,'keys':sorted(d_.keys()),'as_frame':[type(ds.load_iris(as_frame=True).data).__name__,type(ds.load_iris(as_frame=True).target).__name__]}
json.dump(N,open('verify/numbers.json','w'),indent=1); print(json.dumps(N,indent=0))
