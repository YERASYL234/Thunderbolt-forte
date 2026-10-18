from pathlib import Path
import sys,os,json,hashlib,time
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent.parent/'work';DS=ROOT.parent/'forte_dataset'
for p in [WORK/'pydeps',WORK/'scideps',WORK/'train_deps']:
    if p.exists():sys.path.insert(0,str(p))
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np
import torch
import av
from PIL import Image
torch.set_num_threads(4)
SEED=20261017
SAMPLES=24
SPEC={'encoder':'DINOv2 ViT-B/14 frozen','samples':SAMPLES,'fractions':[.01,.99],
      'views':{'full':[0,0,1,1],'work_area':[0,.25,1,.85]},'resize':[224,224],
      'local_features':'2x2 spatial pooling of normalized patch tokens',
      'normalization':'ImageNet mean/std','cache_version':'dino24_dual_v1'}
def videos():return json.loads((DS/'annotations/videos.json').read_text(encoding='utf8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def encoder():
    provenance=json.loads((ROOT/'encoder_provenance.json').read_text(encoding='utf8'))
    sys.path.insert(0,str(ROOT/provenance['vendor_path']))
    from dinov2.hub.backbones import dinov2_vitb14
    m=dinov2_vitb14(pretrained=False)
    state=torch.load(ROOT/'dinov2_vitb14_pretrain.pth',map_location='cpu',weights_only=True)
    m.load_state_dict(state,strict=True);m.eval();m.requires_grad_(False)
    return m
def preprocess(im,box):
    w,h=im.size;im=im.crop((round(box[0]*w),round(box[1]*h),round(box[2]*w),round(box[3]*h)))
    a=np.asarray(im.resize((224,224),Image.Resampling.BICUBIC),dtype=np.float32)/255.
    a=(a-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return torch.from_numpy(a.transpose(2,0,1).copy())
def extract(path,m):
    started=time.monotonic();frames=[];times=[]
    with av.open(str(path)) as c:
        s=c.streams.video[0];dur=float(s.duration*s.time_base) if s.duration else c.duration/1e6
        origin=float((s.start_time or 0)*s.time_base);targets=np.linspace(.01*dur,.99*dur,SAMPLES);pos=0
        for f in c.decode(s):
            if f.pts is None:continue
            t=float(f.pts*s.time_base)-origin
            if pos<SAMPLES and t>=targets[pos]:
                im=f.to_image().convert('RGB');x=torch.stack([preprocess(im,b) for b in SPEC['views'].values()])
                while pos<SAMPLES and t>=targets[pos]:frames.append(x);times.append(t);pos+=1
            if pos==SAMPLES:break
    if len(frames)!=SAMPLES:raise ValueError('Could not sample all frames')
    inputs=torch.stack(frames).reshape(-1,3,224,224);zs=[];patches=[]
    with torch.inference_mode():
        for i in range(0,len(inputs),4):
            output=m.forward_features(inputs[i:i+4]);zs.append(output['x_norm_clstoken'].cpu().numpy())
            local=output['x_norm_patchtokens'].reshape(-1,16,16,768)
            cells=torch.stack([local[:,a:a+8,b:b+8].mean((1,2)) for a in (0,8) for b in (0,8)],dim=1)
            patches.append(cells.cpu().numpy())
    z=np.concatenate(zs).reshape(SAMPLES,2,-1)
    return z,dict(sampled_timestamps_sec=times,elapsed_sec=time.monotonic()-started,duration_sec=dur,spec=SPEC),np.concatenate(patches).reshape(SAMPLES,2,4,768)
def pooled(z,patches=None):
    # Preserve absolute state, motion and order as separate candidate feature families.
    g=z[:,0];r=z[:,1]
    def state(x):return np.concatenate([x.mean(0),x.std(0),x.max(0)])
    def change(x):return np.concatenate([x.std(0),np.abs(np.diff(x,axis=0)).mean(0),x[-4:].mean(0)-x[:4].mean(0)])
    result={'dino_full':state(g),'dino_roi':state(r),'dino_motion':np.concatenate([change(g),change(r)]),
            'dino_order':np.concatenate([chunk.mean(0).reshape(-1) for chunk in np.array_split(z,4)]),
            'dino_dual':np.concatenate([state(g),state(r)])}
    if patches is not None:result['dino_local_motion']=change(patches[:,1].reshape(SAMPLES,-1))
    return result
def matrix(include_dino=True):
    vs=videos();features={}
    for key,suffix in [('resnet','_resnet18_48_fullresize_v1.npz'),('r3d','_r3d18_8x16_fullresize_v1.npz')]:
        features[key]=np.stack([np.load(ROOT.parent/'forte_model/feature_cache'/(v['sha256']+suffix),allow_pickle=False)['features'] for v in vs]).astype(float)
    features['visual_fusion']=np.concatenate([features['resnet'],features['r3d']],axis=1)
    if include_dino:
        rows=[]
        for v in vs:
            with np.load(ROOT/'feature_cache'/(v['sha256']+'_dino24_dual_v1.npz'),allow_pickle=False) as d:rows.append(pooled(d['sequence'],d['patches']))
        for key in rows[0]:features[key]=np.stack([r[key] for r in rows]).astype(float)
        features['dino_r3d']=np.concatenate([features['dino_motion'],features['r3d']],axis=1)
    y=np.array([int(v['source_verdict']=='violation') for v in vs]);folds=np.array([v['provisional_fold'] for v in vs])
    return vs,features,y,folds
if __name__=='__main__':
    cache=ROOT/'feature_cache';cache.mkdir(exist_ok=True);m=None
    (ROOT/'feature_spec.json').write_text(json.dumps(SPEC,indent=2),encoding='utf8')
    for i,v in enumerate(videos()):
        path=DS/v['path'];assert sha(path)==v['sha256'];dest=cache/(v['sha256']+'_dino24_dual_v1.npz')
        if dest.exists():print('CACHED',i+1,v['video_id'],flush=True);continue
        if m is None:m=encoder()
        z,meta,patches=extract(path,m);np.savez_compressed(dest,sequence=z,patches=patches,metadata=json.dumps(meta))
        print('DINO',i+1,'/40',v['video_id'],round(meta['elapsed_sec'],1),'seconds',flush=True)
