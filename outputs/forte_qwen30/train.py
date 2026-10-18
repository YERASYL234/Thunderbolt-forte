"""Train-only nested comparison; test10 is loaded only after a signed selection lock.

Frozen Qwen visual observations + frozen visual embeddings, supervised readouts.
No annotations, filename, duration, hashes, or test labels are predictive features.
"""
from pathlib import Path
import sys,json,argparse,datetime
from observations import ROOT,WORK,np,SPEC_HASH,digest,vectors
sys.path.insert(0,str(ROOT.parent/'forte_improved'))
import data
from search import evaluate,scores
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import joblib

SEED=20261020
VERSION='forte_qwen4_train30_v1'
FEATURE_ROOT=ROOT
OLD_HASH='898f857738c9891cd909544d914ef80a7b0e625efb813422b54ebb43ee150dba'
THRESHOLDS=[.4,.5,.6]
def lr(feature,C):return dict(feature=feature,kind='lr',C=C)
VISUAL=[lr('dino_r3d',.001),lr('dino_motion',.01),lr('resnet',.001)]
CANDIDATES=[
    dict(name='Qwen2B',members=[lr('qwen2',.3)],weights=[1.]),
    *[dict(name=f'Qwen4B_summary_C{c}',members=[lr('qwen4_summary',c)],weights=[1.]) for c in [.03,.3,3.]],
    dict(name='Qwen4B_order',members=[lr('qwen4_order',.03)],weights=[1.]),
    dict(name='Qwen4B_trees',members=[dict(feature='qwen4_summary',kind='trees')],weights=[1.]),
    dict(name='DINO_motion',members=[lr('dino_motion',.001)],weights=[1.]),
    dict(name='R3D18',members=[lr('r3d',.001)],weights=[1.]),
    dict(name='ResNet18',members=[lr('resnet',.001)],weights=[1.]),
    dict(name='Visual_ensemble',members=VISUAL,weights=[1/3]*3),
    dict(name='Qwen4B_visual_fusion',members=[lr('qwen4_summary',.3)]+VISUAL,weights=[.5,1/6,1/6,1/6]),
]

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def save(name,value):(ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
def split():
    p=ROOT.parent/'forte_transfer30/split.json';s=read(p)
    assert len(s['train'])==30 and len(s['test'])==10
    assert {v['sha256'] for v in s['train']}.isdisjoint(v['sha256'] for v in s['test'])
    return s
def plan():
    p=dict(version=VERSION,seed=SEED,feature_spec_hash=SPEC_HASH,candidates=CANDIDATES,thresholds=THRESHOLDS,
        train_count=30,test_count=10,split_sha256=digest(ROOT.parent/'forte_transfer30/split.json'),
        selection='Train30 5-fold CV. Prefer both recall and specificity >= 0.4, then balanced accuracy, macro F1, AUC; no test tuning.',
        validation='Nested 5 outer x 4 inner stratified folds entirely within train30; outer labels excluded from selection.',
        group_unit='Unique video. Session, actor and scene independence cannot be guaranteed.',
        fitting='Frozen Qwen2/4 and visual backbones; supervised classifiers and scalers fitted only on each training fold.',
        test_policy='Lock checkpoint, threshold, prompt and split before extracting test Qwen4 features. Evaluate selected model once.',
        disclosure='All 40 videos appeared in earlier work. Test10 is held out from this fit, not historically untouched.')
    if (ROOT/'training_plan.json').exists():assert read(ROOT/'training_plan.json')==p,'Do not alter registered experiment'
    else:save('training_plan.json',p)
    return p

def load(vs,needed=None):
    needed=set(needed or ['qwen2','qwen4_summary','qwen4_order','r3d','resnet','dino_motion','dino_r3d'])
    features={};q2=[];q4=[]
    for v in vs:
        if 'qwen2' in needed:
            old=read(ROOT.parent/'forte_qwen_trained/feature_cache'/(v['sha256']+'_'+OLD_HASH[:12]+'.json'))
            assert old['spec_hash']==OLD_HASH and old['sha256']==v['sha256']
            states=old['states'];assert len(states)==16 and all(s in ['yes','no','unclear'] for s in states)
            q2.append([float(s==value) for s in states for value in ['yes','no','unclear']])
        if needed & {'qwen4_summary','qwen4_order'}:
            new=read(FEATURE_ROOT/'feature_cache'/(v['sha256']+'_'+SPEC_HASH[:12]+'.json'))
            assert new['spec_hash']==SPEC_HASH and new['sha256']==v['sha256']
            q4.append(vectors(new))
    if q2:features['qwen2']=np.array(q2)
    if q4:
        for k in ['qwen4_summary','qwen4_order']:features[k]=np.stack([r[k] for r in q4])
    for k,suffix in [('r3d','_r3d18_8x16_fullresize_v1.npz'),('resnet','_resnet18_48_fullresize_v1.npz')]:
        if k in needed or (k=='r3d' and 'dino_r3d' in needed):
            features[k]=np.stack([np.load(ROOT.parent/'forte_model/feature_cache'/(v['sha256']+suffix),allow_pickle=False)['features'] for v in vs])
    if needed & {'dino_motion','dino_r3d'}:
        rows=[]
        for v in vs:
            with np.load(ROOT.parent/'forte_improved/feature_cache'/(v['sha256']+'_dino24_dual_v1.npz'),allow_pickle=False) as d:
                rows.append(data.pooled(d['sequence'],d['patches'])['dino_motion'])
        features['dino_motion']=np.stack(rows)
        if 'dino_r3d' in needed:features['dino_r3d']=np.concatenate([features['dino_motion'],features['r3d']],axis=1)
    assert all(np.isfinite(x).all() for x in features.values())
    return features

def estimator(c):
    if c['kind']=='trees':return ExtraTreesClassifier(n_estimators=250,max_depth=3,min_samples_leaf=3,class_weight='balanced',random_state=SEED,n_jobs=2)
    return make_pipeline(StandardScaler(),LogisticRegression(C=c['C'],solver='liblinear',class_weight='balanced',max_iter=2000,random_state=SEED))
def fit(c,X,y):return [estimator(m).fit(X[m['feature']],y) for m in c['members']]
def predict(c,models,X):return np.average([scores(m,X[d['feature']]) for d,m in zip(c['members'],models)],axis=0,weights=c['weights'])
def subset(X,ids):return {k:v[ids] for k,v in X.items()}
def rank(r,t):return (min(r['recall'],r['specificity'])>=.4,r['balanced_accuracy'],r['macro_f1'],r['roc_auc'],-abs(t-.5))

def select(X,y,seed,nfold=5):
    folds=list(StratifiedKFold(nfold,shuffle=True,random_state=seed).split(np.zeros(len(y)),y))
    member_predictions={};rows=[];predictions={}
    for c in CANDIDATES:
        ps=[]
        for member in c['members']:
            key=json.dumps(member,sort_keys=True)
            if key not in member_predictions:
                p=np.zeros(len(y));x=X[member['feature']]
                for a,b in folds:p[b]=scores(estimator(member).fit(x[a],y[a]),x[b])
                member_predictions[key]=p
            ps.append(member_predictions[key])
        p=np.average(ps,axis=0,weights=c['weights']);predictions[c['name']]=p
        choices=[(t,evaluate(y,p,t)) for t in THRESHOLDS];t,r=max(choices,key=lambda item:rank(item[1],item[0]))
        rows.append(dict(candidate=c,threshold=t,metrics=r))
    rows.sort(key=lambda row:rank(row['metrics'],row['threshold']),reverse=True)
    return rows[0],rows,predictions

def train():
    if (ROOT/'selection_locked.json').exists():raise RuntimeError('Selection locked. No retraining/tuning after test.')
    plan();s=split();vs=s['train'];y=np.array([int(v['source_verdict']=='violation') for v in vs]);X=load(vs)
    outer=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(np.zeros(len(y)),y))
    p=np.zeros(len(y));ts=np.zeros(len(y));audit=[]
    with threadpool_limits(limits=2):
        for f,(a,b) in enumerate(outer):
            chosen,_,_=select(subset(X,a),y[a],SEED+f+1,nfold=4)
            c=chosen['candidate'];models=fit(c,subset(X,a),y[a]);p[b]=predict(c,models,subset(X,b));ts[b]=chosen['threshold']
            r=evaluate(y[b],p[b],ts[b][0])
            audit.append(dict(fold=f,train_ids=[vs[i]['video_id'] for i in a],validation_ids=[vs[i]['video_id'] for i in b],selection=chosen,outer_metrics=r))
            print('NESTED',f+1,c['name'],'BA',round(r['balanced_accuracy'],3),flush=True)
        selection,leaderboard,ps=select(X,y,SEED)
        models=fit(selection['candidate'],X,y)
    r=evaluate(y,p-ts+.5);r['roc_auc']=evaluate(y,p)['roc_auc']
    save('nested_metrics.json',dict(metrics=r,folds=audit,scope='Nested development estimate of selection procedure on train30; not independent external validation.'))
    save('nested_predictions.json',[dict(video_id=v['video_id'],label=int(y[i]),score=float(p[i]),threshold=float(ts[i]),prediction=int(p[i]>=ts[i])) for i,v in enumerate(vs)])
    save('development_leaderboard.json',leaderboard)
    save('development_predictions.json',[dict(video_id=v['video_id'],label=int(y[i]),score=float(ps[selection['candidate']['name']][i]),threshold=selection['threshold']) for i,v in enumerate(vs)])
    package=dict(version=VERSION,selection=selection,models=models,training_hashes=[v['sha256'] for v in vs],feature_spec_hash=SPEC_HASH)
    joblib.dump(package,ROOT/'classifier30.joblib',compress=3)
    restored=joblib.load(ROOT/'classifier30.joblib')
    np.testing.assert_allclose(predict(selection['candidate'],models,X),predict(restored['selection']['candidate'],restored['models'],X),atol=1e-12)
    save('split.json',dict(s,model_version=VERSION))
    lock=dict(selection=selection,checkpoint_sha256=digest(ROOT/'classifier30.joblib'),split_sha256=digest(ROOT/'split.json'),
        feature_spec_hash=SPEC_HASH,training_plan_sha256=digest(ROOT/'training_plan.json'),
        locked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),test_qwen4_features_loaded=False,test_predictions_computed=False)
    save('selection_locked.json',lock)
    save('verification.json',dict(status='PASS',training_count=30,test_count=10,hashes_disjoint=True,
        test_used_for_model_selection=False,preprocessing_fitted_within_each_fold=True,checkpoint_reload_matches=True,
        qwen_backbone_finetuned=False,supervised_readout_trained=True,labels_or_ids_in_qwen_prompt=False))
    print('LOCKED',json.dumps(selection),flush=True)
    print('NESTED_RESULT',json.dumps(r),flush=True)

def test():
    if (ROOT/'test_metrics.json').exists():raise RuntimeError('Test already evaluated; preserve results.')
    lock=read(ROOT/'selection_locked.json');s=read(ROOT/'split.json');package=joblib.load(ROOT/'classifier30.joblib')
    assert digest(ROOT/'classifier30.joblib')==lock['checkpoint_sha256']
    assert digest(ROOT/'split.json')==lock['split_sha256']
    assert digest(ROOT/'training_plan.json')==lock['training_plan_sha256']
    assert SPEC_HASH==lock['feature_spec_hash']
    assert set(package['training_hashes'])=={v['sha256'] for v in s['train']}
    assert set(package['training_hashes']).isdisjoint(v['sha256'] for v in s['test'])
    vs=s['test'];y=np.array([int(v['source_verdict']=='violation') for v in vs]);selection=package['selection']
    X=load(vs,{c['feature'] for c in selection['candidate']['members']})
    p=predict(selection['candidate'],package['models'],X);t=selection['threshold'];r=evaluate(y,p,t)
    a=r['accuracy'];n=len(y);z=1.96;den=1+z*z/n;center=(a+z*z/(2*n))/den;half=z*np.sqrt(a*(1-a)/n+z*z/(4*n*n))/den
    report=dict(model_version=VERSION,metrics=r,threshold=t,selection=selection,
        scope='One test of selected model on ten videos excluded from this fit; historically exposed, not a new independent test.',
        accuracy_95_wilson_interval=[center-half,center+half],test_ids=[v['video_id'] for v in vs],
        selection_lock_sha256=digest(ROOT/'selection_locked.json'),tuning_after_test=False)
    save('test_metrics.json',report)
    save('test_predictions.json',[dict(video_id=v['video_id'],true_label=int(y[i]),score=float(p[i]),predicted_label=int(p[i]>=t),correct=bool((p[i]>=t)==y[i])) for i,v in enumerate(vs)])
    print('TEST_RESULT',json.dumps(report),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['plan','train','test'],required=True);a=ap.parse_args()
    {'plan':plan,'train':train,'test':test}[a.stage]()
