"""Prespecified 30/10 transfer-learning experiment, video-level splitting only.

Run --stage train first; --stage test is a single final evaluation of the locked
selection. Frozen pretrained feature caches are reused, never all-40 readouts.
"""
from pathlib import Path
import sys, json, hashlib, shutil, argparse, time, copy
ROOT = Path(__file__).resolve().parent
IMPROVED = ROOT.parent / 'forte_improved'
OLD = ROOT.parent / 'forte_model'
DS = ROOT.parent / 'forte_dataset'
sys.path.insert(0, str(IMPROVED))
import data
from data import np, torch
import finetune as ft
from search import model, scores, evaluate, threshold
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import joblib

SEED = 20261018
VERSION = 'forte_transfer_train30_v1'
FEATURES = ['r3d', 'dino_dual', 'dino_motion', 'dino_r3d', 'resnet']
CANDIDATES = [dict(feature=f, kind=k, C=c) for f in FEATURES
              for k,c in [('lr', .001), ('lr', .01), ('rbf', 1.), ('rbf', 10.)]]
TAIL_EPOCHS = [4, 8, 12]
RECIPE = dict(ft.RECIPE, version='resnet18_tail_train30_v1', epochs=12,
              tail_learning_rate=.00003, head_learning_rate=.0003,
              pooled_layernorm=True, flip_augmentation=False,
              eval_stratified_bags=True,
              model_selection='Epoch count and threshold selected on train30 OOF predictions only')

def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def save(name, obj): (ROOT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf8')
def fingerprint(p): return data.sha(p)
def rank(r): return tuple(r[k] for k in ['balanced_accuracy', 'macro_f1', 'roc_auc'])

def split():
    vs = read(DS/'annotations/videos.json')
    previous = read(ROOT.parent/'forte_r3d_35/split.json')
    old_test = {v['video_id'] for v in previous['test']}
    rng = np.random.default_rng(SEED)
    # Keep the previous five; add two positives and three negatives, once.
    extra = []
    for label, n in [('violation', 2), ('no_violation', 3)]:
        pool = [v['video_id'] for v in vs if v['video_id'] not in old_test and v['source_verdict'] == label]
        extra.extend(rng.choice(pool, n, replace=False).tolist())
    test_ids = old_test | set(extra)
    def entry(v): return {k:v[k] for k in ['video_id','sha256','path','source_names','source_verdict']}
    manifest = dict(model_version=VERSION, seed=SEED,
        method='Keep previous 5 test videos; add 2 positive and 3 negative videos with one fixed random seed; no resampling',
        train=[entry(v) for v in vs if v['video_id'] not in test_ids],
        test=[entry(v) for v in vs if v['video_id'] in test_ids],
        previous_test_ids=sorted(old_test), added_test_ids=sorted(extra),
        training_counts={'violation':16, 'no_violation':14}, test_counts={'violation':5, 'no_violation':5},
        limitation='Held out from this experiment only. All 40 videos were used in historical experiments; people/camera/session independence is not established.')
    assert len(manifest['train']) == 30 and len(manifest['test']) == 10
    assert len({v['sha256'] for v in vs}) == 40
    assert {v['sha256'] for v in manifest['train']}.isdisjoint(v['sha256'] for v in manifest['test'])
    for part in ['train','test']:
        assert sum(v['source_verdict']=='violation' for v in manifest[part]) == manifest['training_counts' if part=='train' else 'test_counts']['violation']
    if (ROOT/'split.json').exists(): assert read(ROOT/'split.json') == manifest
    else: save('split.json', manifest)
    dest = ROOT/'test_videos'; dest.mkdir(exist_ok=True)
    for v in manifest['test']:
        target = dest / Path(v['path']).name
        if not target.exists(): shutil.copyfile(DS/v['path'], target)
        assert fingerprint(target) == v['sha256']
    return manifest

def load_features(vs):
    out = {}
    for key,suffix in [('r3d','_r3d18_8x16_fullresize_v1.npz'), ('resnet','_resnet18_48_fullresize_v1.npz')]:
        out[key] = np.stack([np.load(OLD/'feature_cache'/(v['sha256']+suffix), allow_pickle=False)['features'] for v in vs]).astype(np.float64)
    rows=[]
    for v in vs:
        with np.load(IMPROVED/'feature_cache'/(v['sha256']+'_dino24_dual_v1.npz'),allow_pickle=False) as d:
            rows.append(data.pooled(d['sequence'],d['patches']))
    for key in ['dino_dual','dino_motion']:
        out[key]=np.stack([row[key] for row in rows]).astype(np.float64)
    out['dino_r3d']=np.concatenate([out['dino_motion'],out['r3d']],axis=1)
    assert all(np.isfinite(x).all() for x in out.values())
    return out

def load_maps(vs):
    maps=[]
    for v in vs:
        d=torch.load(IMPROVED/'tail_feature_cache'/(v['sha256']+'_stem160_24x2.pt'),map_location='cpu',weights_only=True)
        assert d['sha256']==v['sha256']
        maps.append(d['maps'])
    return maps

def train_tail(maps,y,train_indices,validation_indices,source,seed,epochs):
    ft.RECIPE=dict(RECIPE)
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    net=ft.TailModel(source).train()
    opt=torch.optim.AdamW([{'params':net.tail.parameters(),'lr':RECIPE['tail_learning_rate']},
                          {'params':list(net.head.parameters())+list(net.norm.parameters()),'lr':RECIPE['head_learning_rate']}],weight_decay=.01)
    count=np.bincount(y[train_indices],minlength=2)
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(count[0]/count[1])))
    predictions={};losses=[];started=time.monotonic()
    for epoch in range(1,epochs+1):
        net.train();order=rng.permutation(train_indices);total=0
        for offset in range(0,len(order),4):
            ids=order[offset:offset+4];batch=[]
            for i in ids:
                take=[rng.integers(j*6,(j+1)*6) for j in range(4)]
                batch.append(maps[i][take].reshape(8,256,10,10).float())
            opt.zero_grad(set_to_none=True)
            loss=criterion(net(torch.stack(batch)),torch.tensor(y[ids],dtype=torch.float32))
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            total+=float(loss.detach())*len(ids)
        losses.append(total/len(order))
        if epoch in TAIL_EPOCHS and len(validation_indices):
            predictions[epoch]=ft.infer(net,maps,validation_indices)
        print('TAIL_EPOCH',epoch,'/',epochs,'loss',round(losses[-1],4),'seconds',round(time.monotonic()-started,1),flush=True)
    delta=float((net.tail[0].conv1.weight.detach()-source.layer4[0].conv1.weight.detach()).abs().max())
    assert delta>0
    return net,predictions,dict(losses=losses,layer4_weight_max_change=delta,elapsed_sec=time.monotonic()-started)

def train():
    if (ROOT/'selection_locked.json').exists(): raise RuntimeError('Selection already locked; do not rerun or tune after test evaluation.')
    manifest=split();vs=manifest['train'];y=np.array([int(v['source_verdict']=='violation') for v in vs])
    plan=dict(seed=SEED,training_videos=30,test_videos=10,candidates=CANDIDATES,
        tail_recipe=RECIPE,tail_epoch_options=TAIL_EPOCHS,threshold_options=[.35,.5,.65],
        validation='Stratified 5-fold by unique video, entirely inside train30; same folds for every candidate',
        selection_objective=['balanced_accuracy','macro_f1','roc_auc'],
        ensemble='One equal-weight ensemble of the best three distinct feature families is also considered',
        test_policy='Load test embeddings only after checkpoint and selection are locked. Evaluate selected model once; no tuning afterward.',
        label_inputs='Binary source_verdict only. No filenames, video IDs, event text or point labels used as predictive features.',
        learned_components='StandardScaler + supervised readouts; separate real ImageNet ResNet18 layer4 and MIL head fine-tuning',
        sources=['https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r3d_18.html',
                 'https://github.com/facebookresearch/dinov2',
                 'https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html'])
    save('training_plan.json',plan)
    folds=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(np.zeros(len(y)),y))
    save('fold_audit.json',[dict(fold=f,train_ids=[vs[i]['video_id'] for i in a],validation_ids=[vs[i]['video_id'] for i in b]) for f,(a,b) in enumerate(folds)])
    print('TRAIN_IDS',','.join(v['video_id'] for v in vs),flush=True)
    print('TEST_IDS',','.join(v['video_id'] for v in manifest['test']),flush=True)
    features=load_features(vs);all_p={};rows=[]
    def record(name,c,p):
        t,r=threshold(y,p);all_p[name]=p
        rows.append(dict(name=name,candidate=c,threshold=t,metrics=r))
        print('CANDIDATE',name,'BA',round(r['balanced_accuracy'],3),'F1',round(r['f1'],3),'T',t,flush=True)
    with threadpool_limits(limits=2):
        for i,c in enumerate(CANDIDATES):
            p=np.zeros(len(y));x=features[c['feature']]
            for a,b in folds:
                clf=model(c).fit(x[a],y[a]);p[b]=scores(clf,x[b])
            record('readout_'+str(i),c,p)
    maps=load_maps(vs);source=ft.base();tail_p={e:np.zeros(len(y)) for e in TAIL_EPOCHS};audits=[]
    for f,(a,b) in enumerate(folds):
        print('TAIL_FOLD',f+1,'/5',flush=True)
        net,p,audit=train_tail(maps,y,a,b,source,SEED+f,max(TAIL_EPOCHS))
        for e in TAIL_EPOCHS:tail_p[e][b]=p[e]
        audits.append(dict(fold=f,training=audit));del net
    for e,p in tail_p.items():record('tail_'+str(e),dict(feature='resnet_tail',kind='finetune',epochs=e),p)
    rows.sort(key=lambda r:rank(r['metrics']),reverse=True)
    distinct=[];seen=set()
    for r in rows:
        family=r['candidate']['feature']
        if family not in seen:distinct.append(r);seen.add(family)
        if len(distinct)==3:break
    ep=np.mean([all_p[r['name']] for r in distinct],axis=0);et,er=threshold(y,ep)
    best=rows[0]
    if rank(er)>rank(best['metrics']):
        chosen=distinct;selected_p=ep;selected_t=et;selected_r=er
    else:chosen=[best];selected_p=all_p[best['name']];selected_t=best['threshold'];selected_r=best['metrics']
    selection=dict(model_version=VERSION,threshold=selected_t,members=chosen,
        development_metrics=selected_r,ensemble=len(chosen)>1,
        development_metrics_scope='Selection CV on train30; optimistic because candidates/threshold were selected using these predictions. Not final test.',
        chosen_without_test=True)
    save('development_leaderboard.json',rows)
    save('ensemble_candidate.json',dict(members=[r['name'] for r in distinct],threshold=et,metrics=er))
    save('development_predictions.json',[dict(video_id=v['video_id'],label=int(y[i]),score=float(selected_p[i]),threshold=selected_t) for i,v in enumerate(vs)])
    # Train a real fine-tuned alternative on all 30 even if readout wins selection.
    best_tail=next(r for r in rows if r['candidate']['kind']=='finetune')
    selected_tail=next((r for r in chosen if r['candidate']['kind']=='finetune'),best_tail)
    epochs=selected_tail['candidate']['epochs'];print('FINAL_TAIL_FIT',epochs,flush=True)
    tail,_,audit=train_tail(maps,y,np.arange(len(y)),[],source,SEED+100,epochs)
    recipe=dict(RECIPE,epochs=epochs)
    torch.save(dict(state_dict=tail.state_dict(),recipe=recipe,training_hashes=[v['sha256'] for v in vs]),ROOT/'finetuned_tail30.pt')
    restored=ft.TailModel(source);restored.load_state_dict(torch.load(ROOT/'finetuned_tail30.pt',map_location='cpu',weights_only=True)['state_dict'])
    np.testing.assert_allclose(ft.infer(tail,maps,[0,1]),ft.infer(restored,maps,[0,1]),atol=1e-7)
    save('finetuning_audit.json',dict(folds=audits,final=audit,recipe=recipe,checkpoint_reload_matches=True))
    models=[]
    for row in chosen:
        c=row['candidate']
        models.append(None if c['kind']=='finetune' else model(c).fit(features[c['feature']],y))
    package=dict(model_version=VERSION,selection=selection,models=models,training_hashes=[v['sha256'] for v in vs],feature_spec=data.SPEC)
    joblib.dump(package,ROOT/'classifier30.joblib',compress=3)
    reloaded=joblib.load(ROOT/'classifier30.joblib')
    for row,m,m2 in zip(chosen,models,reloaded['models']):
        if m is not None:
            np.testing.assert_allclose(scores(m,features[row['candidate']['feature']]),scores(m2,features[row['candidate']['feature']]),atol=1e-12)
    lock=dict(selection=selection,checkpoint_sha256=fingerprint(ROOT/'classifier30.joblib'),tail_checkpoint_sha256=fingerprint(ROOT/'finetuned_tail30.pt'),
              split_sha256=fingerprint(ROOT/'split.json'),locked_at_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
              test_embeddings_loaded=False,test_predictions_computed=False)
    save('selection_locked.json',lock)
    save('verification.json',dict(status='PASS',training_count=30,test_count=10,hashes_disjoint=True,
        previous_five_test_preserved=True,fit_preprocessing_on_fold_train_only=True,test_used_for_model_selection=False,
        old_all40_readouts_loaded=False,real_backbone_finetuning=True,checkpoint_reload_matches=True))
    print('LOCKED_SELECTION',json.dumps(selection),flush=True)

def test():
    if (ROOT/'test_metrics.json').exists(): raise RuntimeError('Final test already evaluated; use saved results, do not rerun selection.')
    lock=read(ROOT/'selection_locked.json');manifest=read(ROOT/'split.json')
    assert fingerprint(ROOT/'classifier30.joblib')==lock['checkpoint_sha256']
    assert fingerprint(ROOT/'finetuned_tail30.pt')==lock['tail_checkpoint_sha256']
    assert fingerprint(ROOT/'split.json')==lock['split_sha256']
    package=joblib.load(ROOT/'classifier30.joblib');vs=manifest['test'];y=np.array([int(v['source_verdict']=='violation') for v in vs])
    assert set(package['training_hashes'])=={v['sha256'] for v in manifest['train']}
    assert set(package['training_hashes']).isdisjoint(v['sha256'] for v in vs)
    features=load_features(vs);ps=[]
    for row,m in zip(package['selection']['members'],package['models']):
        c=row['candidate']
        if c['kind']=='finetune':
            checkpoint=torch.load(ROOT/'finetuned_tail30.pt',map_location='cpu',weights_only=True)
            ft.RECIPE=checkpoint['recipe'];net=ft.TailModel(ft.base());net.load_state_dict(checkpoint['state_dict'])
            ps.append(ft.infer(net,load_maps(vs),np.arange(len(vs))))
        else:ps.append(scores(m,features[c['feature']]))
    p=np.mean(ps,axis=0);t=package['selection']['threshold'];r=evaluate(y,p,t)
    n=len(y);a=r['accuracy'];z=1.96;den=1+z*z/n;center=(a+z*z/(2*n))/den
    half=z*np.sqrt(a*(1-a)/n+z*z/(4*n*n))/den
    report=dict(model_version=VERSION,metrics=r,threshold=t,
        evaluation='One evaluation on ten videos excluded from this experiment training and model selection; historically exposed data, not a new independent external test.',
        descriptive_accuracy_95_wilson_interval=[center-half,center+half],test_ids=[v['video_id'] for v in vs],
        selection_lock_sha256=fingerprint(ROOT/'selection_locked.json'),tuning_after_test=False)
    save('test_predictions.json',[dict(video_id=v['video_id'],sha256=v['sha256'],true_label=int(y[i]),score=float(p[i]),threshold=t,predicted_label=int(p[i]>=t),correct=bool((p[i]>=t)==y[i])) for i,v in enumerate(vs)])
    save('test_metrics.json',report)
    print('FINAL_TEST',json.dumps(report),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['train','test'],required=True);args=ap.parse_args()
    train() if args.stage=='train' else test()
