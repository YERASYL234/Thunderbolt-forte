from features import *
import argparse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('video');ap.add_argument('--output',default=str(ROOT/'prediction.json'));ap.add_argument('--gpu-layers',type=int,default=99);ap.add_argument('--cached-features');a=ap.parse_args()
    cfg=json.loads((ROOT/'config.json').read_text(encoding='utf8'))
    if cfg['feature_spec_hash']!=SPEC_HASH:raise ValueError('Feature specification differs from training')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    if a.cached_features:
        d=json.loads(Path(a.cached_features).read_text(encoding='utf8'))
        if d['sha256']!=digest(a.video) or d['spec_hash']!=SPEC_HASH:raise ValueError('Cached features do not match video/spec')
    else:
        with LocalQwen(out.with_suffix('.server.log'),a.gpu_layers) as m:d=m.extract(a.video)
    with np.load(ROOT/'classifier.npz',allow_pickle=False) as w:
        if str(w['spec_hash'])!=SPEC_HASH:raise ValueError('Checkpoint feature mismatch')
        logit=float((vector(d['states'])@w['coef'].T+w['intercept']).item())
    score=float(1/(1+np.exp(-np.clip(logit,-50,50))))
    report=json.loads((ROOT/'metrics.json').read_text(encoding='utf8'))
    seen=any(v['sha256']==d['sha256'] for v in report['training_data'])
    r=dict(model_version=cfg['model_version'],video=str(Path(a.video).resolve()),prediction='violation' if score>=.5 else 'no_violation',
           score=score,score_is_calibrated_probability=False,video_seen_in_final_training=seen,
           qwen_weights_updated=False,trained_component=cfg['trained_component'],observations=d,
           limitation='Experimental source-label prediction; not a verified violation. No event class or localization. Use OOF metrics, not this final-model prediction, to assess development performance.')
    out.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(r,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
