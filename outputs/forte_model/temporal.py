"""Exploratory temporal transfer-learning model; entirely local after weights download."""
from model import *
from torchvision.models.video import r3d_18,R3D_18_Weights
from train import pipeline,metrics
import socket

def video_encoder(download=False):
    torch.set_num_threads(4);p=ROOT/'r3d18_kinetics400.pth'
    if not p.exists():
        if not download:raise FileNotFoundError('Local r3d18 weights missing')
        url=R3D_18_Weights.KINETICS400_V1.url
        socket.setdefaulttimeout(90)
        torch.hub.download_url_to_file(url,str(p),hash_prefix=url.rsplit('-',1)[1].split('.')[0],progress=False)
    m=r3d_18(weights=None);m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True));m.fc=torch.nn.Identity();m.eval();return m

def video_features(path,m):
    started=time.perf_counter();c=av.open(str(path));s=c.streams.video[0]
    dur=float(s.duration*s.time_base) if s.duration else c.duration/1e6;origin=float((s.start_time or 0)*s.time_base)
    fps=float(s.average_rate or 30);last=max(0,dur-1/fps);span=min(2.,last)
    starts=np.linspace(0,max(0,last-span),8)
    requests=sorted([(float(t),k,j) for k,a in enumerate(starts) for j,t in enumerate(np.linspace(a,a+span,16))])
    clips=[[None]*16 for _ in range(8)];actual=[[None]*16 for _ in range(8)];pos=0;latest=None;time_last=None
    for f in c.decode(s):
        if f.pts is None:continue
        t=float(f.pts*s.time_base)-origin
        if pos<len(requests) and t>=requests[pos][0]:
            a=np.asarray(f.to_image().resize((112,112),Image.Resampling.BILINEAR),dtype=np.float32)/255.
            a=(a-np.array([.43216,.394666,.37645],np.float32))/np.array([.22803,.22145,.216989],np.float32)
            latest=torch.from_numpy(a.transpose(2,0,1).copy());time_last=t
            while pos<len(requests) and t>=requests[pos][0]:
                _,k,j=requests[pos];clips[k][j]=latest;actual[k][j]=t;pos+=1
        if pos==len(requests):break
    c.close()
    if latest is None:raise ValueError('Video yielded no sampleable frames')
    for _,k,j in requests[pos:]:clips[k][j]=latest;actual[k][j]=time_last
    zs=[]
    with torch.inference_mode():
        for clip in clips:
            z=m(torch.stack(clip,dim=1).unsqueeze(0)).numpy()[0];zs.append(z)
    z=np.stack(zs);features=np.concatenate([z.mean(0),z.std(0),z.max(0)]).astype(np.float32)
    return features,dict(duration_sec=dur,clip_timestamps_sec=actual,elapsed_sec=time.perf_counter()-started)

def run_training(ds):
    vs=json.loads((ds/'annotations/videos.json').read_text(encoding='utf8'));m=video_encoder(download=True);cache=ROOT/'feature_cache';cache.mkdir(exist_ok=True)
    xs=[];sampling=[]
    for i,v in enumerate(vs):
        p=ds/v['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256']
        fp=cache/(v['sha256']+'_r3d18_8x16_fullresize_v1.npz')
        if fp.exists():d=np.load(fp,allow_pickle=False);x=d['features'];meta=json.loads(str(d['metadata']))
        else:x,meta=video_features(p,m);np.savez_compressed(fp,features=x,metadata=json.dumps(meta))
        xs.append(x);sampling.append(dict(video_id=v['video_id'],**meta));print(f'TEMPORAL {i+1}/40 {v["video_id"]} {meta["elapsed_sec"]:.1f}s',flush=True)
    X=np.stack(xs);y=np.array([v['source_verdict']=='violation' for v in vs],int);folds=np.array([v['provisional_fold'] for v in vs]);prob=np.zeros(len(vs))
    for f in range(5):
        tr=folds!=f;te=~tr;clf=pipeline().fit(X[tr],y[tr]);prob[te]=clf.predict_proba(X[te])[:,1]
    clf=pipeline().fit(X,y)
    np.savez_compressed(ROOT/'temporal_classifier.npz',mean=clf[0].mean_,scale=clf[0].scale_,coef=clf[1].coef_,intercept=clf[1].intercept_)
    cfg=dict(model_version='forte_binary_r3d18_v1',encoder='frozen torchvision R3D18 KINETICS400_V1',trained_component='L2 logistic readout',clips_per_video=8,frames_per_clip=16,clip_span_sec=2.,spatial_resize=[112,112],decision_threshold=.5,C=.01,event_model=False,encoder_sha256=hashlib.sha256((ROOT/'r3d18_kinetics400.pth').read_bytes()).hexdigest())
    report=dict(config=cfg,metrics=metrics(y,prob),training_fit_metrics_NOT_generalization=metrics(y,clf.predict_proba(X)[:,1]),evaluation='Exploratory 5-fold development evaluation on same folds as image baseline. Compared after seeing baseline results; NOT independent final test.',limitations=['40 unique videos; similar scenes/actors across folds.','Eight sampled windows do not cover every moment; brief events may be missed.','Frozen Kinetics action encoder; binary supervised readout only.','No violation type, exact timestamp or legal grounding.','No 95% accuracy claim.'],feature_extraction_seconds=dict(mean=float(np.mean([s['elapsed_sec'] for s in sampling])),max=float(max(s['elapsed_sec'] for s in sampling))))
    predictions=[dict(video_id=v['video_id'],fold=int(folds[i]),true_label=int(y[i]),predicted_label=int(prob[i]>=.5),score=float(prob[i]),correct=bool(y[i]==(prob[i]>=.5))) for i,v in enumerate(vs)]
    np.testing.assert_allclose([score(x,np.load(ROOT/'temporal_classifier.npz',allow_pickle=False)) for x in X],clf.predict_proba(X)[:,1],atol=1e-7,rtol=1e-6)
    for name,obj in [('temporal_config.json',cfg),('temporal_metrics.json',report),('temporal_oof_predictions.json',predictions),('temporal_sampling_audit.json',sampling)]: (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
    print('TEMPORAL_OUT_OF_FOLD',json.dumps(report['metrics']),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--dataset',default=str(ROOT.parent/'forte_dataset'));ap.add_argument('--video');ap.add_argument('--output');a=ap.parse_args()
    if a.train:run_training(Path(a.dataset));return
    if not a.video:ap.error('Specify --train or --video')
    m=video_encoder();x,meta=video_features(a.video,m);p=score(x,np.load(ROOT/'temporal_classifier.npz',allow_pickle=False))
    result=dict(model_version='forte_binary_r3d18_v1',verdict='violation' if p>=.5 else 'no_violation',violation_score=p,score_is_calibrated_probability=False,processing_sec=meta['elapsed_sec'],event_localization_supported=False,limitation='Experimental binary classifier; not reliable for operational use.')
    text=json.dumps(result,ensure_ascii=False,indent=2);print(text)
    if a.output:Path(a.output).write_text(text,encoding='utf8')
if __name__=='__main__':main()
