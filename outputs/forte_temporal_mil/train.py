from common import *
from model import TemporalMIL,make_windows,point_mask,normalizer,loss_for_bag,event_candidates
import time,copy,math
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix

def metrics(y,p,threshold=.5):
    y=np.array(y);p=np.array(p);pred=p>=threshold
    return dict(n=len(y),accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred,zero_division=0)),
        f1=float(f1_score(y,pred,zero_division=0)),roc_auc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,
        confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist(),threshold=threshold)

def load_data():
    vs=videos();seq=[];bounds=[];raw_windows=[];points={v['video_id']:[] for v in vs}
    for e in read(DS/'annotations/events.json'):
        t=e.get('source_timecode_sec')
        if isinstance(t,(int,float)):points[e['video_id']].append(float(t))
    for v in vs:
        z=np.load(ROOT/'embeddings'/(v['video_id']+'.npy'),allow_pickle=False)
        meta=read(ROOT/'embeddings'/(v['video_id']+'.json'))
        assert meta['video_sha256']==v['sha256'] and meta['spec']==SPEC
        assert np.isfinite(z).all() and z.shape[1:]==(2,384)
        assert all(0<=p<=meta['duration_sec'] for p in points[v['video_id']])
        seq.append(torch.from_numpy(z.reshape(len(z),768)))
        w,b=make_windows(z,meta['actual_timestamps_sec'],meta['duration_sec']);raw_windows.append(w);bounds.append(b)
    y=np.array([int(v['source_verdict']=='violation') for v in vs]);folds=np.array([v['provisional_fold'] for v in vs])
    assert set(folds)==set(range(5))
    assert all(len(set(y[folds==f]))==2 for f in range(5))
    return vs,seq,raw_windows,bounds,points,y,folds

@torch.inference_mode()
def evaluate(model,windows,indices,y):
    model.eval();scores=[];all_windows=[];losses=[]
    for i in indices:
        logits=model(windows[i]);video=model.pool(logits)
        scores.append(float(video.sigmoid()));all_windows.append(logits.sigmoid().numpy().tolist())
        losses.append(float(torch.nn.functional.binary_cross_entropy_with_logits(video,video.new_tensor(float(y[i])))))
    return np.array(scores),all_windows,float(np.mean(losses))

def fit(indices,vs,seq,raw,bounds,points,y,seed,validation=None,epochs=None,tag='fit'):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);model=TemporalMIL()
    mean,std=normalizer(seq,indices);windows=[(w-mean[None,:,None])/std[None,:,None] for w in raw]
    masks={i:point_mask(bounds[i],points[vs[i]['video_id']],CONFIG['point_radius_sec']) if y[i] else torch.zeros(len(bounds[i]),dtype=torch.bool) for i in indices}
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    counts=np.bincount(y[indices],minlength=2);class_weights=len(indices)/(2*np.maximum(counts,1))
    history=[];best=float('inf');best_epoch=CONFIG['min_epochs'];best_state=None;initial=model.head.weight.detach().clone();initial_conv=model.temporal[0].weight.detach().clone()
    stop=epochs or CONFIG['max_epochs'];started=time.monotonic()
    for epoch in range(1,stop+1):
        model.train();order=rng.permutation(indices).tolist();losses=[];terms=[]
        for start in range(0,len(order),CONFIG['batch_size']):
            batch=order[start:start+CONFIG['batch_size']];sizes=[len(windows[i]) for i in batch]
            logits=model(torch.cat([windows[i] for i in batch])).split(sizes)
            batch_losses=[]
            for i,l in zip(batch,logits):
                loss,term=loss_for_bag(model,l,int(y[i]),masks[i],class_weights);batch_losses.append(loss);terms.append(term.numpy())
            loss=torch.stack(batch_losses).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite MIL loss')
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
        row=dict(epoch=epoch,train_loss=float(np.mean(losses)),loss_terms=np.mean(terms,axis=0).tolist())
        if validation is not None:
            _,_,vl=evaluate(model,windows,validation,y);row['validation_video_bce']=vl
            # Keep the best checkpoint from epoch 1; min_epochs only gates stopping.
            # Otherwise an early good model would be discarded after overfitting.
            if vl<best-1e-5:
                best=vl;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
        history.append(row)
        if epoch==1 or epoch%10==0:print(tag,'epoch',epoch,'loss',round(row['train_loss'],4),'val',round(row.get('validation_video_bce',0),4),'seconds',round(time.monotonic()-started,1),flush=True)
        if validation is not None and epoch>=CONFIG['min_epochs'] and epoch-best_epoch>=CONFIG['patience']:break
    if validation is not None:
        assert best_state is not None;model.load_state_dict(best_state)
    else:best_epoch=stop
    delta=float((model.head.weight.detach()-initial).abs().max());conv_delta=float((model.temporal[0].weight.detach()-initial_conv).abs().max());assert delta>0 and conv_delta>0
    pack=dict(state_dict=model.state_dict(),mean=mean,std=std,config=CONFIG,feature_spec=SPEC,epochs=best_epoch,
              seed=seed,training_video_ids=[vs[i]['video_id'] for i in indices],training_hashes=[vs[i]['sha256'] for i in indices],
              point_supervision_video_ids=[vs[i]['video_id'] for i in indices if y[i]],head_max_weight_change=delta,conv_max_weight_change=conv_delta)
    return model,windows,pack,history

def main():
    vs,seq,raw,bounds,points,y,folds=load_data();assert len(set(v['sha256'] for v in vs))==len(vs)
    plan=dict(config=CONFIG,feature_spec=SPEC,fold_assignment={v['video_id']:int(folds[i]) for i,v in enumerate(vs)},
        grouping='Source video SHA256; every window of a video stays in one fold. Not actor/session-disjoint.',
        early_stopping='For each outer fold, one other fold is inner validation; retain best video-BCE checkpoint from epoch 1. Run at least 30 and at most 60 inner epochs, patience 10; refit outer training for selected best epoch count, which may be below 30.',
        threshold='Fixed 0.5 before evaluation; not tuned on outer predictions.',
        final_fit='All 40 videos; epoch count is median inner-selected epochs.',
        supervision='Positive timestamps used only for training videos; +-1 s around window centers, nearest window fallback. Unlabeled positive-video regions are not negative.',
        limitations=['Source point timestamps and binary labels remain provisional.','Same development dataset used in earlier experiments.','No independent final test or verified event intervals.'])
    save(ROOT/'training_plan.json',plan);oof=np.zeros(len(vs));records=[];audit=[];histories={};selected_epochs=[]
    checkpoints=ROOT/'fold_models';checkpoints.mkdir(exist_ok=True)
    for f in range(5):
        te=np.flatnonzero(folds==f);outer=np.flatnonzero(folds!=f);inner_fold=(f+1)%5
        tr=outer[folds[outer]!=inner_fold];va=outer[folds[outer]==inner_fold]
        assert set(tr).isdisjoint(te) and set(va).isdisjoint(te) and set(tr).isdisjoint(va)
        assert len(set(y[tr]))==len(set(y[va]))==len(set(y[te]))==2
        _,_,stopping,h1=fit(tr,vs,seq,raw,bounds,points,y,SEED+f,validation=va,tag=f'fold{f}_inner')
        ep=stopping['epochs'];selected_epochs.append(ep)
        model,windows,pack,h2=fit(outer,vs,seq,raw,bounds,points,y,SEED+100+f,epochs=ep,tag=f'fold{f}_refit')
        assert set(pack['training_hashes']).isdisjoint(vs[i]['sha256'] for i in te)
        scores,ws,_=evaluate(model,windows,te,y);oof[te]=scores
        torch.save(pack,checkpoints/f'fold_{f}.pt')
        for i,p,w in zip(te,scores,ws):
            records.append(dict(video_id=vs[i]['video_id'],sha256=vs[i]['sha256'],fold=f,true_label=int(y[i]),score=float(p),
                predicted_label=int(p>=.5),correct=bool((p>=.5)==y[i]),window_bounds_sec=bounds[i].tolist(),window_scores=w,
                candidate_events=event_candidates(bounds[i],w)))
        audit.append(dict(outer_fold=f,inner_validation_fold=inner_fold,inner_train_ids=[vs[i]['video_id'] for i in tr],
            inner_validation_ids=[vs[i]['video_id'] for i in va],outer_train_ids=pack['training_video_ids'],outer_test_ids=[vs[i]['video_id'] for i in te],
            point_supervision_video_ids=pack['point_supervision_video_ids'],selected_epochs=ep,metrics=metrics(y[te],scores)))
        histories[str(f)]=dict(inner=h1,outer_refit=h2)
        save(ROOT/'fold_audit.json',audit);save(ROOT/'oof_predictions.json',records);save(ROOT/'training_history.json',histories)
        print('FOLD_DONE',f,json.dumps(audit[-1]['metrics']),flush=True)
    epochs=int(np.median(selected_epochs));all_indices=np.arange(len(vs))
    model,windows,pack,h=fit(all_indices,vs,seq,raw,bounds,points,y,SEED+999,epochs=epochs,tag='final_all40')
    pack.update(encoder_sha256=read(ROOT/'encoder_provenance.json')['weights_sha256'],evaluation='Development grouped 5-fold; no independent final test')
    torch.save(pack,ROOT/'temporal_mil.pt');histories['final']=h;save(ROOT/'training_history.json',histories)
    restored=TemporalMIL(pack['config']);restored.load_state_dict(torch.load(ROOT/'temporal_mil.pt',weights_only=True)['state_dict'])
    model.eval();restored.eval()
    with torch.inference_mode():torch.testing.assert_close(model(windows[0]),restored(windows[0]),rtol=0,atol=0)
    result=dict(development_metrics=metrics(y,oof),selected_epochs=selected_epochs,final_epochs=epochs,model_parameters=sum(p.numel() for p in model.parameters()),
                architecture='Frozen DINOv2 ViT-S/14 + dual-view 2 FPS + 4s 1D-CNN + top-10% MIL',plan=plan)
    save(ROOT/'metrics.json',result);print('MIL_RESULT',json.dumps(result['development_metrics']),flush=True)
if __name__=='__main__':main()
