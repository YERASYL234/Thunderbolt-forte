from common import *
from model import TemporalMIL,make_windows,event_candidates
from extract import extract_video
import argparse,time

def predict(video_path,checkpoint=None):
    started=time.monotonic();path=Path(video_path).resolve();digest=sha(path)
    cp=ROOT/'temporal_mil.pt' if checkpoint is None else Path(checkpoint)
    pack=torch.load(cp,map_location='cpu',weights_only=True)
    assert pack['feature_spec']==SPEC
    match=next((v for v in videos() if v['sha256']==digest),None)
    cached=ROOT/'embeddings'/((match['video_id'] if match else digest)+'.npy')
    meta=read(cached.with_suffix('.json')) if match and cached.exists() and cached.with_suffix('.json').exists() else None
    if meta and meta['spec']==SPEC and meta['video_sha256']==digest:
        z=np.load(cached,allow_pickle=False)
    else:z,meta=extract_video(path,encoder(),progress=True)
    windows,bounds=make_windows(z,meta['actual_timestamps_sec'],meta['duration_sec'],pack['config'])
    windows=(windows-pack['mean'][None,:,None])/pack['std'][None,:,None]
    model=TemporalMIL(pack['config']);model.load_state_dict(pack['state_dict']);model.eval()
    with torch.inference_mode():
        logits=model(windows);score=float(model.pool(logits).sigmoid());scores=logits.sigmoid().numpy()
    return dict(video_path=str(path),video_sha256=digest,model=str(cp),video_score=score,threshold=pack['config']['threshold'],
        predicted_label='violation' if score>=pack['config']['threshold'] else 'no_violation',
        candidate_events=event_candidates(bounds,scores,pack['config']['threshold']),
        window_bounds_sec=bounds.tolist(),window_scores=scores.tolist(),
        seen_in_training=digest in pack['training_hashes'],experimental=True,
        note='Uncalibrated model scores; candidate windows are not verified violations or exact event boundaries. Event type not trained.',
        elapsed_sec=time.monotonic()-started)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('--checkpoint');p.add_argument('--output');args=p.parse_args()
    result=predict(args.video,args.checkpoint)
    if args.output:save(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['window_bounds_sec','window_scores']},ensure_ascii=False,indent=2))
