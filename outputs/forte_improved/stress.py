"""Exploratory transfer across automatically inferred background groups."""
from data import *
from search import candidates,select,train_selection,selection_scores,evaluate
from threadpoolctl import threadpool_limits
vs,features,y,folds=matrix(True)
group_data=json.loads((ROOT/'background_groups.json').read_text(encoding='utf8'))
lookup={r['video_id']:r['background_group'] for r in group_data['videos']}
groups=np.array([lookup[v['video_id']] for v in vs]);p=np.zeros(len(y));pred=np.zeros(len(y),int);details=[]
for g in sorted(set(groups)):
    tr=groups!=g;te=~tr;cs=candidates(features)
    # PCA(8) requires at least 8 samples in every inner training split.
    if min(int(np.sum(tr & (folds!=f))) for f in set(folds[tr]))<8:cs=[c for c in cs if c['kind']!='pca_lr']
    print('BACKGROUND_HOLDOUT',int(g),'train',int(tr.sum()),'test',int(te.sum()),flush=True)
    with threadpool_limits(limits=2):
        selected,_=select({k:x[tr] for k,x in features.items()},y[tr],folds[tr],cs)
        models=train_selection(selected,{k:x[tr] for k,x in features.items()},y[tr])
        p[te]=selection_scores(selected,models,{k:x[te] for k,x in features.items()})
    pred[te]=p[te]>=selected['threshold']
    details.append(dict(group=int(g),train_count=int(tr.sum()),test_count=int(te.sum()),selection=selected,
                        metrics=evaluate(y[te],p[te],selected['threshold']),test_videos=[v['video_id'] for v,b in zip(vs,te) if b]))
    print('BACKGROUND_RESULT',json.dumps(details[-1]['metrics']),flush=True)
r=evaluate(y,p)
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
r.update(accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
         precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred,zero_division=0)),
         specificity=float(recall_score(1-y,1-pred,zero_division=0)),f1=float(f1_score(y,pred,zero_division=0)),
         macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())
(ROOT/'background_stress.json').write_text(json.dumps(dict(metrics=r,groups=details,
    warning='Exploratory background-cluster transfer, not independently verified camera/session split. One direction trains on only 8 videos. No independent final test.'),indent=2),encoding='utf8')
