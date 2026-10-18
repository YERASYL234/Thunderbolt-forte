from common import *
import time,argparse
import av
from PIL import Image

def preprocess(im,box):
    w,h=im.size
    im=im.crop(tuple(round(c*d) for c,d in zip(box,[w,h,w,h])))
    a=np.asarray(im.resize((224,224),Image.Resampling.BICUBIC),dtype=np.float32)/255
    a=(a-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return torch.from_numpy(a.transpose(2,0,1).copy())

def extract_video(path,m,progress=False):
    started=time.monotonic();pending=[];vectors=[];times=[];requested=[];last_report=started
    def flush():
        if not pending:return
        with torch.inference_mode():
            z=m.forward_features(torch.stack(pending))['x_norm_clstoken'].cpu().numpy()
        vectors.append(z);pending.clear()
    with av.open(str(path)) as c:
        stream=c.streams.video[0]
        duration=float(stream.duration*stream.time_base) if stream.duration else c.duration/1e6
        origin=float((stream.start_time or 0)*stream.time_base);target=0.0
        for frame in c.decode(stream):
            if frame.pts is None:continue
            t=float(frame.pts*stream.time_base)-origin
            if t+1e-6<target:continue
            im=frame.to_image().convert('RGB')
            # At most one target per source frame for these ~30 FPS sources.
            views=[preprocess(im,box) for box in SPEC['views'].values()]
            while target<=t+1e-6:
                pending.extend(views);times.append(t);requested.append(target);target+=1/SPEC['fps']
                if len(pending)>=16:flush()
            if progress and time.monotonic()-last_report>35:
                print('  frames',len(times),'video_sec',round(t,1),flush=True);last_report=time.monotonic()
    flush()
    if not vectors:raise ValueError('No frames decoded')
    z=np.concatenate(vectors).reshape(-1,2,384).astype(np.float32)
    assert len(z)==len(times) and np.isfinite(z).all()
    meta={'spec':SPEC,'actual_timestamps_sec':times,'requested_timestamps_sec':requested,
          'duration_sec':duration,'elapsed_sec':time.monotonic()-started,'shape':list(z.shape)}
    return z,meta

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);args=parser.parse_args()
    dest=ROOT/'embeddings';dest.mkdir(exist_ok=True);save(ROOT/'feature_spec.json',SPEC);model=None
    for i,v in enumerate(videos()[:args.limit]):
        target=dest/(v['video_id']+'.npy');metadata=target.with_suffix('.json')
        if target.exists() and metadata.exists():
            old=read(metadata)
            if old.get('video_sha256')==v['sha256'] and old['spec']==SPEC:
                print('CACHED',v['video_id'],flush=True);continue
        path=DS/v['path'];assert sha(path)==v['sha256']
        if model is None:model=encoder()
        z,meta=extract_video(path,model,progress=True)
        meta.update(video_id=v['video_id'],video_sha256=v['sha256'],encoder_sha256=read(ROOT/'encoder_provenance.json')['weights_sha256'])
        with target.with_suffix('.tmp').open('wb') as f:np.save(f,z,allow_pickle=False)
        target.with_suffix('.tmp').replace(target);save(metadata,meta)
        print('EMBEDDING',i+1,'/40',v['video_id'],list(z.shape),round(meta['elapsed_sec'],1),'sec',flush=True)
if __name__=='__main__':main()
