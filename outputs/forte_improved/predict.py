"""Run the selected local video classifier, with explicit experimental status."""
from data import *
from search import selection_scores
import argparse,joblib

def visual_features(path,needed):
    out={};need_r3d=any(k in needed for k in ('r3d','visual_fusion','dino_r3d'))
    need_resnet=any(k in needed for k in ('resnet','visual_fusion'))
    if any(k.startswith('dino') for k in needed):
        m=encoder();z,meta,patch=extract(path,m);out.update(pooled(z,patch));del m
    if need_r3d or need_resnet:
        sys.path.insert(0,str(ROOT.parent/'forte_model'))
        import model as old
        if need_resnet:
            m=old.encoder(old.ROOT/'resnet18_imagenet.pth');out['resnet'],_=old.extract(path,m,48);del m
        if need_r3d:
            import temporal
            m=temporal.video_encoder();out['r3d'],_=temporal.video_features(path,m);del m
    if 'visual_fusion' in needed:out['visual_fusion']=np.concatenate([out['resnet'],out['r3d']])
    if 'dino_r3d' in needed:out['dino_r3d']=np.concatenate([out['dino_motion'],out['r3d']])
    return {k:out[k][None,:] for k in needed}
def pipeline_prediction(path):
    p=joblib.load(ROOT/'all_classifier.joblib')
    features=visual_features(path,{c['feature'] for c in p['selection']['members']})
    return float(selection_scores(p['selection'],p['models'],features)[0]),float(p['selection']['threshold'])
def tail_prediction(path,stable=False,matched=False):
    import finetune as ft
    state=torch.load(ROOT/('finetuned_matched_tail.pt' if matched else ('finetuned_stable_tail.pt' if stable else 'finetuned_tail.pt')),map_location='cpu',weights_only=True)
    ft.RECIPE=state['recipe'];base=ft.base();maps,_=ft.sample_maps(path,ft.early(base));m=ft.TailModel(base)
    m.load_state_dict(state['state_dict']);return float(ft.infer(m,[maps],[0])[0]),.5
def old_prediction(path):
    sys.path.insert(0,str(ROOT.parent/'forte_model'));import temporal
    m=temporal.video_encoder();x,_=temporal.video_features(path,m)
    return temporal.score(x,np.load(ROOT.parent/'forte_model/temporal_classifier.npz',allow_pickle=False)),.5
def window_prediction(path):
    import window_model as wm
    package=joblib.load(ROOT/'window_classifier.joblib');m=encoder();z,meta,patch=extract(path,m)
    bags,bounds=wm.window_features(z,patch,meta['sampled_timestamps_sec']);scores=wm.window_scores(package['model'],bags)
    p=wm.aggregate(scores,package['selection']['aggregation']);top=np.argsort(scores)[-3:][::-1]
    windows=[dict(start_sample_sec=float(bounds[i,0]),end_sample_sec=float(bounds[i,1]),score=float(scores[i]),
                  status='candidate_window_not_verified_event_boundary') for i in top]
    return p,float(package['selection']['threshold']),windows
def main():
    ap=argparse.ArgumentParser();ap.add_argument('video');ap.add_argument('--output',default=str(ROOT/'prediction.json'))
    ap.add_argument('--model',choices=['selected','dino_pipeline','finetune','finetune_stable','finetune_matched','ensemble','r3d','window','window_r3d'],default='selected');a=ap.parse_args()
    started=time.monotonic();path=Path(a.video).resolve();chosen=a.model
    if chosen=='selected':chosen=json.loads((ROOT/'selection.json').read_text(encoding='utf8'))['selected_model']
    candidates=[]
    if chosen=='dino_pipeline':score,t=pipeline_prediction(path)
    elif chosen=='finetune':score,t=tail_prediction(path)
    elif chosen=='finetune_stable':score,t=tail_prediction(path,True)
    elif chosen=='finetune_matched':score,t=tail_prediction(path,matched=True)
    elif chosen=='ensemble':
        p,_=pipeline_prediction(path);q,_=tail_prediction(path);score=(p+q)/2;t=.5
    elif chosen=='window':score,t,candidates=window_prediction(path)
    elif chosen=='window_r3d':
        p,_,candidates=window_prediction(path);q,_=old_prediction(path);score=(p+q)/2;t=.5
    else:score,t=old_prediction(path)
    fingerprint=sha(path);seen=any(v['sha256']==fingerprint for v in videos())
    result=dict(model=chosen,video=str(path),score=score,threshold=t,prediction='violation' if score>=t else 'no_violation',
                processing_sec=time.monotonic()-started,video_seen_in_training=seen,score_is_calibrated_probability=False,
                event_class_supported=False,operational_use_validated=False,
                candidate_windows=candidates,
                warning='Experimental video-level source-label prediction. No independent final-test accuracy; a prediction is not proof of a violation.')
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
