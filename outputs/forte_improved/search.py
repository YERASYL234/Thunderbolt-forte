"""Nested model/threshold selection; outer labels never enter model selection."""
from data import *
import argparse, collections, math, joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
from threadpoolctl import threadpool_limits

def model(c):
    steps=[StandardScaler()]
    if c['kind']=='pca_lr':steps.append(PCA(n_components=8,svd_solver='full'))
    if c['kind'] in ('lr','pca_lr'):steps.append(LogisticRegression(C=c['C'],solver='liblinear',class_weight='balanced',max_iter=2000,random_state=SEED))
    else:steps.append(SVC(C=c['C'],kernel='rbf',gamma='scale',class_weight='balanced'))
    return make_pipeline(*steps)
def scores(m,X):
    if hasattr(m,'predict_proba'):return m.predict_proba(X)[:,1]
    return 1/(1+np.exp(-np.clip(m.decision_function(X),-50,50)))
def evaluate(y,p,threshold=.5):
    pred=p>=threshold
    return dict(n=int(len(y)),accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred,zero_division=0)),
        specificity=float(recall_score(1-y,1-pred.astype(int),zero_division=0)),f1=float(f1_score(y,pred,zero_division=0)),
        macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),roc_auc=float(roc_auc_score(y,p)),
        confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())
def candidates(features):
    return [dict(feature=f,kind=k,C=c) for f in features for k,cs in [('lr',[.001,.01,.1,1.]),('rbf',[.1,1.,10.]),('pca_lr',[.1,1.])] for c in cs]
def threshold(y,p):
    # This function is called ONLY on inner-validation predictions.
    options=[]
    for t in [.35,.5,.65]:
        r=evaluate(y,p,t);options.append((r['balanced_accuracy'],r['macro_f1'],-abs(t-.5),t,r))
    best=max(options,key=lambda x:x[:3]);return best[3],best[4]
def select(features,y,folds,cs):
    ps=[];ranked=[]
    for i,c in enumerate(cs):
        p=np.zeros(len(y));X=features[c['feature']]
        for fold in sorted(set(folds)):
            tr=folds!=fold;te=~tr
            assert len(set(y[tr]))==2
            clf=model(c).fit(X[tr],y[tr]);p[te]=scores(clf,X[te])
        t,r=threshold(y,p);ps.append(p);ranked.append(dict(index=i,threshold=t,metrics=r,candidate=c))
    ranked.sort(key=lambda r:(r['metrics']['balanced_accuracy'],r['metrics']['macro_f1'],r['metrics']['roc_auc']),reverse=True)
    # Equal-weight ensemble of up to 3 different feature families, selected using inner data only.
    chosen=[];seen=set()
    for r in ranked:
        f=r['candidate']['feature']
        if f not in seen:chosen.append(r['index']);seen.add(f)
        if len(chosen)==3:break
    ep=np.mean([ps[i] for i in chosen],axis=0);et,er=threshold(y,ep)
    best=ranked[0];ens_key=(er['balanced_accuracy'],er['macro_f1'],er['roc_auc'])
    best_key=tuple(best['metrics'][k] for k in ['balanced_accuracy','macro_f1','roc_auc'])
    if len(chosen)>1 and ens_key>best_key:
        selection=dict(members=[cs[i] for i in chosen],threshold=et,inner_metrics=er,ensemble=True)
    else:selection=dict(members=[best['candidate']],threshold=best['threshold'],inner_metrics=best['metrics'],ensemble=False)
    return selection,ranked
def train_selection(selection,features,y):
    return [model(c).fit(features[c['feature']],y) for c in selection['members']]
def selection_scores(selection,models,features):
    return np.mean([scores(m,features[c['feature']]) for c,m in zip(selection['members'],models)],axis=0)
def nested(features,y,folds,cs):
    p=np.zeros(len(y));pred=np.zeros(len(y),int);thresholds=np.zeros(len(y));details=[]
    for fold in sorted(set(folds)):
        tr=folds!=fold;te=~tr
        selection,_=select({k:X[tr] for k,X in features.items()},y[tr],folds[tr],cs)
        models=train_selection(selection,{k:X[tr] for k,X in features.items()},y[tr])
        p[te]=selection_scores(selection,models,{k:X[te] for k,X in features.items()})
        thresholds[te]=selection['threshold'];pred[te]=(p[te]>=selection['threshold']).astype(int)
        detail=dict(fold=int(fold),train_indices=np.flatnonzero(tr).tolist(),test_indices=np.flatnonzero(te).tolist(),selection=selection,
                    metrics=evaluate(y[te],p[te],selection['threshold']))
        details.append(detail)
        print('OUTER_FOLD',int(fold),'accuracy',round(detail['metrics']['accuracy'],3),'selected',selection['members'],flush=True)
    r=evaluate(y,p)
    # A fold-specific inner-selected threshold is used for classification; AUC still uses raw scores.
    r.update(accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
             precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred,zero_division=0)),
             specificity=float(recall_score(1-y,1-pred,zero_division=0)),f1=float(f1_score(y,pred,zero_division=0)),
             macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())
    return r,p,pred,thresholds,details
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--existing-only',action='store_true');args=ap.parse_args()
    vs,features,y,folds=matrix(not args.existing_only);cs=candidates(features)
    suffix='existing' if args.existing_only else 'all'
    plan=dict(candidate_count=len(cs),candidates=cs,thresholds=[.35,.5,.65],primary_objective='balanced_accuracy',
              selection='inner folds only; equal-weight top-3 feature-family ensemble also evaluated inside inner folds',
              outer_folds='existing provisional 5-fold groups',seed=SEED)
    (ROOT/(suffix+'_search_plan.json')).write_text(json.dumps(plan,indent=2),encoding='utf8')
    print('NESTED_SEARCH',suffix,len(cs),'candidates',flush=True)
    with threadpool_limits(limits=2):
        r,p,pred,thresholds,details=nested(features,y,folds,cs)
        selection,ranked=select(features,y,folds,cs)
        models=train_selection(selection,features,y)
    package=dict(version='forte_improved_'+suffix,selection=selection,models=models,feature_spec=SPEC,
                 trained_components='supervised readouts; image/video backbones frozen',training_hashes=[v['sha256'] for v in vs])
    joblib.dump(package,ROOT/(suffix+'_classifier.joblib'),compress=3)
    restored=joblib.load(ROOT/(suffix+'_classifier.joblib'))
    np.testing.assert_allclose(selection_scores(selection,models,features),selection_scores(restored['selection'],restored['models'],features),atol=1e-12)
    majority=np.zeros(len(y))
    for f in sorted(set(folds)):majority[folds==f]=int(y[folds!=f].mean()>=.5)
    n=len(y);a=r['accuracy'];z=1.96;den=1+z*z/n;c=(a+z*z/(2*n))/den;h=z*math.sqrt(a*(1-a)/n+z*z/(4*n*n))/den
    report=dict(evaluation='Nested 5-fold DEVELOPMENT evaluation of the model-selection procedure; NOT independent final test',
                nested_metrics=r,accuracy_interval_descriptive_only=[c-h,c+h],final_selection=selection,
                final_selection_inner_metrics_NOT_test=selection['inner_metrics'],candidate_count=len(cs),outer_folds=details,
                majority_baseline=evaluate(y,majority),trained_components=package['trained_components'],
                limitations=['40 unique videos only; labels not independently adjudicated.','Existing folds were seen in previous experiments; repeated development is not a new final test.',
                             'Actors, cameras and scenarios can overlap across folds.','Readouts trained; pretrained encoders frozen.','No 95% claim. No event type or time localization.','Scores uncalibrated.'])
    rows=[dict(video_id=v['video_id'],sha256=v['sha256'],fold=int(folds[i]),true_label=int(y[i]),predicted_label=int(pred[i]),
               score=float(p[i]),threshold=float(thresholds[i]),correct=bool(pred[i]==y[i])) for i,v in enumerate(vs)]
    for name,obj in [('metrics',report),('oof_predictions',rows),('development_leaderboard',ranked)]:
        (ROOT/(suffix+'_'+name+'.json')).write_text(json.dumps(obj,indent=2),encoding='utf8')
    print('NESTED_RESULT',suffix,json.dumps(r),flush=True)
if __name__=='__main__':main()
