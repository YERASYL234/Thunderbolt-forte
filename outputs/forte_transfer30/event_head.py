"""Weakly supervised event windows and three broad event groups, train30 only.

Point annotations are training targets, never an inference input. Window bounds
are sampled-frame spans, not verified event starts/ends or proof of a bank rule.
"""
from pathlib import Path
import sys,json,math
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'forte_improved'))
from data import np,DS,ROOT as FEATURE_ROOT,sha
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,f1_score,precision_score,recall_score,confusion_matrix
import joblib

VERSION='forte_event_windows_train30_v1'
UNKNOWN='suspicious_action_untyped'
GROUPS={
 'cash_placement':{'name':'Ақшаны орналастыру немесе қалдыру','description':'Ақша не ақша салынған қапты орналастыру/қалдыруға ұқсас әрекет','review':'Ақшаның қараусыз қалғанын және операция жағдайын тексеру қажет.'},
 'key_handling':{'name':'Кілтпен әрекет','description':'Кілтті алып кету немесе құлыпта қалдыруға ұқсас әрекет','review':'Кілттің көрінуін, қайда қойылғанын және рұқсат етілген тәртіпті тексеру қажет.'},
 'customer_procedure':{'name':'Клиентке қызмет көрсету тәртібі','description':'Клиентпен операция кезіндегі әрекетке ұқсас көрініс','review':'Операцияның толық ретін тексеру қажет; қандай міндетті әрекет орындалмағаны автоматты расталмайды.'},
}
TYPE_THRESHOLD=.65
TYPE_MARGIN=.10
PARAMETERS=dict(C=.01,seed=20261019,window_samples=3,stride_samples=1,
    detector_threshold_candidates=[.5,.65,.8,.9],type_threshold=TYPE_THRESHOLD,type_margin=TYPE_MARGIN,
    max_events=3,minimum_distinct_positive_videos=3)

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(name,obj):(ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')

def window_features(sequence,times,duration):
    z=np.asarray(sequence);times=np.asarray(times,dtype=float)
    assert z.ndim==3 and z.shape[1:]==(2,768) and len(z)==len(times)
    assert len(z)>0 and np.isfinite(z).all() and np.isfinite(times).all()
    assert np.all(np.diff(times)>=0)
    if len(z)<3:
        z=np.concatenate([z,np.repeat(z[-1:],3-len(z),axis=0)])
        times=np.pad(times,(0,3-len(times)),mode='edge')
    x=[];bounds=[];centers=[]
    for start in range(len(z)-2):
        part=z[start:start+3]
        x.append(np.concatenate([part.mean(0).ravel(),(part[-1]-part[0]).ravel()]))
        bounds.append([float(np.clip(times[start],0,duration)),float(np.clip(times[start+2],0,duration))])
        centers.append(float(np.clip(times[start+1],0,duration)))
    return np.stack(x).astype(np.float64),np.array(bounds),np.array(centers)

def fit_head(videos,bags,points,indices,group=None):
    xx=[];yy=[];ww=[];audit=[]
    for i in indices:
        v=videos[i];x,bounds,centers=bags[i];label=v['source_verdict']=='violation'
        # Type discrimination is conditional on an event: compare labeled event
        # groups to one another. Normal windows belong only to the event detector.
        if group is not None and not label:continue
        if label:
            # Unannotated regions of positive videos are NOT assumed to be normal.
            tagged={}
            for point in points.get(v['video_id'],[]):
                index=int(np.argmin(np.abs(centers-point['source_timecode_sec'])))
                tagged.setdefault(index,set()).add(point['parent_class'])
            take=sorted(tagged)
            if group is not None:take=[k for k in take if tagged[k].intersection(GROUPS)]
            if group is not None and not take:continue
            if not take:raise ValueError('Missing training timestamp '+v['video_id'])
            labels=[int(group is None or group in tagged[k]) for k in take]
        else:take=list(range(len(x)));labels=[0]*len(take)
        xx.extend(x[take]);yy.extend(labels);ww.extend([1/len(take)]*len(take))
        audit.append(dict(video_id=v['video_id'],window_indices=take,targets=labels))
    xx=np.stack(xx);yy=np.array(yy);ww=np.array(ww)
    assert set(yy)=={0,1},'Both training classes required'
    # Equal total video weight before class balancing; repeated normal windows do
    # not count as independent extra videos.
    for value in [0,1]:ww[yy==value]*=len(indices)/(2*ww[yy==value].sum())
    scaler=StandardScaler().fit(xx,sample_weight=ww)
    clf=LogisticRegression(C=PARAMETERS['C'],solver='liblinear',max_iter=2000,random_state=PARAMETERS['seed'])
    clf.fit(scaler.transform(xx),yy,sample_weight=ww)
    return dict(scaler=scaler,classifier=clf,training_audit=audit)

def score(head,x):return head['classifier'].predict_proba(head['scaler'].transform(x))[:,1]

def choose_type(group_scores):
    ranked=sorted(group_scores.items(),key=lambda p:p[1],reverse=True)
    top,confidence=ranked[0];runner=ranked[1][1] if len(ranked)>1 else 0.
    accepted=confidence>=TYPE_THRESHOLD and confidence-runner>=TYPE_MARGIN
    return (top if accepted else UNKNOWN),ranked

def events_from_scores(event_scores,type_scores,bounds,centers,binary_positive,detector_threshold):
    """A positive binary alert never implies a verified class or event boundary."""
    if not binary_positive:
        return dict(status='video_negative',events=[],message='Видео моделі бұзушылық белгісін таппады. Бұл бұзушылық жоқтығына кепілдік емес.')
    order=np.argsort(-event_scores,kind='stable');chosen=[]
    qualifying=[int(i) for i in order if event_scores[i]>=detector_threshold]
    # If the detector cannot support the binary alert, retain one ranked window
    # as a review candidate with an explicit unresolved class, never force a type.
    for i in qualifying or [int(order[0])]:
        a,b=bounds[i]
        if any(max(0,min(b,bounds[j,1])-max(a,bounds[j,0])) / max(1e-9,min(b-a,bounds[j,1]-bounds[j,0]))>.5 for j in chosen):continue
        chosen.append(i)
        if len(chosen)==PARAMETERS['max_events']:break
    events=[]
    for rank,i in enumerate(chosen,1):
        values={group:float(p[i]) for group,p in type_scores.items()}
        group,ranked=choose_type(values)
        if event_scores[i]<detector_threshold:group=UNKNOWN
        typed=group!=UNKNOWN
        events.append(dict(event_id=f'prediction-{rank}',class_id=group,
            class_name=GROUPS[group]['name'] if typed else 'Типі анықталмаған күмәнді әрекет',
            description=GROUPS[group]['description'] if typed else 'Видео моделі сигнал берді, бірақ оқиға түрін сенімді ажырату мүмкін болмады.',
            status='candidate_type' if typed else 'type_unresolved',
            source='event_window_model',event_score=float(event_scores[i]),
            event_threshold=float(detector_threshold),class_score=values[group] if typed else None,
            score_is_calibrated_probability=False,
            timestamp_sec=float(centers[i]),window_start_sec=float(bounds[i,0]),window_end_sec=float(bounds[i,1]),
            interval_status='sampled_window_not_verified_event_boundary',verified_rule_violation=False,rule_id=None,
            verification_needed=GROUPS[group]['review'] if typed else 'Кадрлар мен толық видеоны тексеру қажет; сигнал жалған дабыл болуы мүмкін.',
            alternatives=[dict(class_id=k,class_name=GROUPS[k]['name'],score=float(v),is_selected=typed and k==group) for k,v in ranked]))
    status='candidate_types' if any(e['status']=='candidate_type' for e in events) else 'type_unresolved'
    return dict(status=status,events=events,message='Модель ұсынған оқиға түрлері мен уақыт аралықтары. Бұзушылық расталды дегенді білдірмейді.' if status=='candidate_types' else 'Күмәнді сигнал бар, бірақ оқиға түрі нақтыланбады. Ең жоғары ұпайлы үзінді тексеруге көрсетілді; жалған дабыл болуы мүмкін.')

class EventHead:
    def __init__(self):
        self.package=joblib.load(ROOT/'event_head30.joblib')
        split=read(ROOT/'split.json')
        assert set(self.package['training_hashes'])=={v['sha256'] for v in split['train']}
        assert set(self.package['training_hashes']).isdisjoint(v['sha256'] for v in split['test'])

    def predict(self,sequence,times,duration,binary_positive):
        x,bounds,centers=window_features(sequence,times,duration)
        event_scores=score(self.package['detector'],x)
        type_scores={k:score(v,x) for k,v in self.package['types'].items()}
        result=events_from_scores(event_scores,type_scores,bounds,centers,binary_positive,self.package['threshold'])
        result.update(model_version=VERSION,supported_classes=[dict(class_id=k,**v) for k,v in GROUPS.items()],
                      granularity='broad_event_group',training_video_count=30,
                      event_classification_supported=True,verified_rule_detection_supported=False,
                      validation_status='weak_point_supervision_only_not_independently_validated')
        return result

def train():
    split=read(ROOT/'split.json');videos=split['train'];train_ids={v['video_id'] for v in videos}
    # Filter annotations before selecting training windows. Test event text/timecodes
    # and test features are never passed to fit or threshold selection.
    points={v['video_id']:[] for v in videos}
    for e in read(DS/'annotations/events.json'):
        if e['video_id'] in train_ids and isinstance(e.get('source_timecode_sec'),(float,int)):points[e['video_id']].append(e)
    counts={k:len({v for v,es in points.items() if any(e['parent_class']==k for e in es)}) for k in GROUPS}
    assert all(n>=PARAMETERS['minimum_distinct_positive_videos'] for n in counts.values())
    plan=dict(version=VERSION,parameters=PARAMETERS,groups=GROUPS,positive_video_counts=counts,
        split_sha256=sha(ROOT/'split.json'),supervision='Weak point timestamp and broad parent_class from training videos only',
        type_supervision='Conditional group classifiers fit only supported labeled event windows; event detector separately learns normal versus event windows',
        limits=['Three broad event groups, not 14 reliably learned specific violations.',
                'Time windows from 24 uniformly sampled frames; short events can be missed.',
                'Type scores and detector scores are uncalibrated.',
                'Point labels are not verified interval or rule ground truth.',
                'Existing test10 not used to fit, choose threshold, or report new event accuracy.'])
    save('event_head_plan.json',plan)
    bags=[]
    for v in videos:
        with np.load(FEATURE_ROOT/'feature_cache'/(v['sha256']+'_dino24_dual_v1.npz'),allow_pickle=False) as d:
            meta=json.loads(str(d['metadata']));bags.append(window_features(d['sequence'],meta['sampled_timestamps_sec'],meta['duration_sec']))
    y=np.array([int(v['source_verdict']=='violation') for v in videos]);index={v['video_id']:i for i,v in enumerate(videos)}
    oof=np.zeros(len(y));type_true=[];type_pred=[];fold_audit=[]
    for f in read(ROOT/'fold_audit.json'):
        a=[index[v] for v in f['train_ids']];b=[index[v] for v in f['validation_ids']]
        detector=fit_head(videos,bags,points,a);types={k:fit_head(videos,bags,points,a,k) for k in GROUPS}
        for i in b:
            x,_,centers=bags[i];oof[i]=float(score(detector,x).max())
            ps={k:score(head,x) for k,head in types.items()}
            for e in points[videos[i]['video_id']]:
                if e['parent_class'] not in GROUPS:continue
                j=int(np.argmin(np.abs(centers-e['source_timecode_sec'])))
                group,_=choose_type({k:float(p[j]) for k,p in ps.items()})
                type_true.append(e['parent_class']);type_pred.append(group)
        fold_audit.append(dict(fold=f['fold'],train_ids=f['train_ids'],validation_ids=f['validation_ids']))
    candidates=[]
    for t in PARAMETERS['detector_threshold_candidates']:
        pred=oof>=t
        candidates.append(dict(threshold=t,balanced_accuracy=float(balanced_accuracy_score(y,pred)),f1=float(f1_score(y,pred)),
            specificity=float(recall_score(1-y,1-pred.astype(int),zero_division=0)),confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist()))
    best=max(candidates,key=lambda r:(r['balanced_accuracy'],r['specificity'],r['f1'],r['threshold']))
    final=dict(version=VERSION,detector=fit_head(videos,bags,points,list(range(30))),
        types={k:fit_head(videos,bags,points,list(range(30)),k) for k in GROUPS},threshold=best['threshold'],
        training_hashes=[v['sha256'] for v in videos],plan=plan)
    joblib.dump(final,ROOT/'event_head30.joblib',compress=3)
    loaded=joblib.load(ROOT/'event_head30.joblib')
    np.testing.assert_allclose(score(final['detector'],bags[0][0]),score(loaded['detector'],bags[0][0]),atol=1e-12)
    accepted=sum(p!=UNKNOWN for p in type_pred);correct=sum(t==p for t,p in zip(type_true,type_pred))
    report=dict(version=VERSION,threshold_selection_cv=best,threshold_options=candidates,folds=fold_audit,
        type_diagnostic=dict(label_count=len(type_true),accepted_count=accepted,correct_count=correct,
                            recall_including_abstentions=correct/max(1,len(type_true)),
                            note='Diagnostic uses known held-out point positions; NOT end-to-end localization/classification accuracy.'),
        test10_evaluated=False,training_video_count=30,positive_video_counts=counts,
        checkpoint_sha256=sha(ROOT/'event_head30.joblib'),checkpoint_reload_matches=True,
        test_features_or_timestamps_used_for_fit=False)
    save('event_head_training.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':train()
