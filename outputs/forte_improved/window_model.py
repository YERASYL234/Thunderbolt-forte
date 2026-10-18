"""Weak timestamp-supervised windows; all evaluation is on whole held-out videos."""
from data import *
from search import evaluate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
import joblib

def window_features(z,patches,times):
    starts=list(range(0,len(z)-2,2))
    if starts[-1]!=len(z)-3:starts.append(len(z)-3)
    result={'roi':[],'local':[]};bounds=[]
    for s in starts:
        for name,x in [('roi',z[s:s+3,1]),('local',patches[s:s+3,1].reshape(3,-1))]:
            result[name].append(np.concatenate([x.mean(0),x[-1]-x[0]]))
        bounds.append([float(times[s]),float(times[s+2])])
    return {k:np.stack(v).astype(float) for k,v in result.items()},np.array(bounds)
def load_data():
    vs=videos();bags=[];bounds=[];points={v['video_id']:[] for v in vs}
    for e in json.loads((DS/'annotations/events.json').read_text(encoding='utf8')):
        t=e.get('source_timecode_sec')
        if isinstance(t,(int,float)):points[e['video_id']].append(float(t))
    for v in vs:
        with np.load(ROOT/'feature_cache'/(v['sha256']+'_dino24_dual_v1.npz'),allow_pickle=False) as d:
            meta=json.loads(str(d['metadata']));b,w=window_features(d['sequence'],d['patches'],meta['sampled_timestamps_sec'])
        bags.append(b);bounds.append(w)
    y=np.array([int(v['source_verdict']=='violation') for v in vs]);folds=np.array([v['provisional_fold'] for v in vs])
    return vs,bags,bounds,points,y,folds
def fit(c,indices,vs,bags,bounds,points,y):
    X=[];Y=[];W=[];audit=[];counts=np.bincount(y[indices],minlength=2)
    for i in indices:
        if y[i]:
            # Only training-video point annotations are read here. They remain weak, unverified supervision.
            ts=sorted(set(points[vs[i]['video_id']]))
            if not ts:raise ValueError('Positive training video has no timestamp')
            centers=bounds[i].mean(1);take=sorted({int(np.abs(centers-t).argmin()) for t in ts})
        else:take=list(range(len(bounds[i])))
        X.extend(bags[i][c['feature']][take]);Y.extend([y[i]]*len(take))
        W.extend([len(indices)/(2*counts[y[i]]*len(take))]*len(take))
        audit.append(dict(video_id=vs[i]['video_id'],weak_positive=bool(y[i]),selected_windows=take))
    X=np.stack(X);Y=np.asarray(Y);W=np.asarray(W)
    scaler=StandardScaler().fit(X,sample_weight=W);X=scaler.transform(X)
    clf=LogisticRegression(C=c['C'],solver='liblinear',max_iter=2000,random_state=SEED) if c['kind']=='lr' else SVC(C=c['C'],gamma='scale',kernel='rbf')
    clf.fit(X,Y,sample_weight=W)
    return dict(scaler=scaler,classifier=clf,candidate=c,training_window_audit=audit)
def window_scores(m,bag):
    x=m['scaler'].transform(bag[m['candidate']['feature']]);clf=m['classifier']
    if hasattr(clf,'predict_proba'):return clf.predict_proba(x)[:,1]
    return 1/(1+np.exp(-np.clip(clf.decision_function(x),-50,50)))
def aggregate(p,mode):return float(np.max(p) if mode=='max' else np.sort(p)[-3:].mean())
def candidates():return [dict(feature=f,kind=k,C=c) for f in ['roi','local'] for k,cs in [('lr',[.01,.1,1.,10.]),('rbf',[.1,1.,10.])] for c in cs]
def select(indices,vs,bags,bounds,points,y,folds):
    ranked=[]
    for c in candidates():
        predictions={}
        for f in sorted(set(folds[indices])):
            tr=indices[folds[indices]!=f];te=indices[folds[indices]==f]
            m=fit(c,tr,vs,bags,bounds,points,y)
            for i in te:predictions[int(i)]=window_scores(m,bags[i])
        for mode in ['max','top3mean']:
            p=np.array([aggregate(predictions[int(i)],mode) for i in indices])
            for t in [.5,.65,.8]:
                r=evaluate(y[indices],p,t)
                ranked.append(dict(candidate=c,aggregation=mode,threshold=t,inner_metrics=r))
    ranked.sort(key=lambda r:tuple(r['inner_metrics'][k] for k in ['balanced_accuracy','macro_f1','roc_auc']),reverse=True)
    return ranked[0],ranked
def main():
    vs,bags,bounds,points,y,folds=load_data();p=np.zeros(len(y));pred=np.zeros(len(y),int);thresholds=np.zeros(len(y));details=[]
    plan=dict(candidates=candidates(),aggregations=['max','top3mean'],thresholds=[.5,.65,.8],
              supervision='Only training-video user point timestamps used; positive clips weakly labeled, unknown positive-video regions not labeled negative',
              evaluation='All label-independent windows of held-out videos; held-out timestamps never used for selection or inference')
    (ROOT/'window_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf8')
    with threadpool_limits(limits=2):
        for f in sorted(set(folds)):
            tr=np.flatnonzero(folds!=f);te=np.flatnonzero(folds==f);selection,_=select(tr,vs,bags,bounds,points,y,folds)
            m=fit(selection['candidate'],tr,vs,bags,bounds,points,y)
            assert set(vs[i]['video_id'] for i in te).isdisjoint(a['video_id'] for a in m['training_window_audit'])
            for i in te:p[i]=aggregate(window_scores(m,bags[i]),selection['aggregation'])
            thresholds[te]=selection['threshold'];pred[te]=p[te]>=selection['threshold']
            details.append(dict(fold=int(f),selection=selection,metrics=evaluate(y[te],p[te],selection['threshold']),training_window_audit=m['training_window_audit']))
            print('WINDOW_FOLD',int(f),json.dumps(details[-1]['metrics']),flush=True)
        selection,ranked=select(np.arange(len(y)),vs,bags,bounds,points,y,folds)
        final=fit(selection['candidate'],np.arange(len(y)),vs,bags,bounds,points,y)
    joblib.dump(dict(model=final,selection=selection,training_hashes=[v['sha256'] for v in vs]),ROOT/'window_classifier.joblib',compress=3)
    restored=joblib.load(ROOT/'window_classifier.joblib');np.testing.assert_allclose(window_scores(final,bags[0]),window_scores(restored['model'],bags[0]),atol=1e-12)
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
    r=evaluate(y,p);r.update(accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),precision=float(precision_score(y,pred,zero_division=0)),
        recall=float(recall_score(y,pred,zero_division=0)),specificity=float(recall_score(1-y,1-pred,zero_division=0)),f1=float(f1_score(y,pred,zero_division=0)),macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())
    report=dict(nested_metrics=r,final_selection=selection,outer_folds=details,plan=plan,
        evaluation='Nested 5-fold DEVELOPMENT evaluation; no independent test',
        limitations=['Point timestamps are unverified user annotations, not gold event intervals.','24 sampled frames can miss short events.','Same small development dataset reused.','Only window classifier trained; DINO frozen.','Candidate windows are not confirmed event boundaries.'])
    rows=[dict(video_id=v['video_id'],sha256=v['sha256'],fold=int(folds[i]),true_label=int(y[i]),score=float(p[i]),threshold=float(thresholds[i]),predicted_label=int(pred[i]),correct=bool(pred[i]==y[i])) for i,v in enumerate(vs)]
    for name,obj in [('window_metrics',report),('window_oof_predictions',rows),('window_development_leaderboard',ranked)]:
        (ROOT/(name+'.json')).write_text(json.dumps(obj,indent=2),encoding='utf8')
    print('WINDOW_RESULT',json.dumps(r),flush=True)
if __name__=='__main__':main()
