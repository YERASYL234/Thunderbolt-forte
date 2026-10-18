from data import *
from sklearn.tree import DecisionTreeClassifier
from search import evaluate
vs=videos();X=np.asarray([v['duration_sec'] for v in vs]).reshape(-1,1)
y=np.asarray([int(v['source_verdict']=='violation') for v in vs]);folds=np.asarray([v['provisional_fold'] for v in vs]);p=np.zeros(len(vs));thresholds=[]
for f in sorted(set(folds)):
    tr=folds!=f;te=~tr;m=DecisionTreeClassifier(max_depth=1,class_weight='balanced',random_state=SEED).fit(X[tr],y[tr]);p[te]=m.predict_proba(X[te])[:,1]
    thresholds.append(dict(fold=int(f),duration_cutoff_seconds=float(m.tree_.threshold[0])))
r=dict(metrics=evaluate(y,p),thresholds=thresholds,
    warning='Shortcut diagnostic only: this classifier sees video duration, not pixels or actions. It is not a violation detector and is excluded from deployment selection.')
(ROOT/'duration_shortcut_diagnostic.json').write_text(json.dumps(r,indent=2),encoding='utf8');print(json.dumps(r,indent=2))
