"""Local binary-video baseline. Frame encoder is frozen; video readout is trained."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
workspace=ROOT.parent.parent
for p in [workspace/'work/pydeps', workspace/'work/scideps', workspace/'work/train_deps']:
    if p.exists(): sys.path.insert(0,str(p))
import argparse,json,time,hashlib
import numpy as np
import torch
from torchvision.models import resnet18
from PIL import Image
import av

def encoder(weights,threads=4):
    torch.set_num_threads(threads)
    m=resnet18(weights=None)
    m.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    m.fc=torch.nn.Identity();m.eval()
    return m

def tensor(im):
    # Full field of view: no label-dependent crop, no event timestamps.
    arr=np.asarray(im.resize((224,224),Image.Resampling.BILINEAR),dtype=np.float32)/255
    arr=(arr-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
    return torch.from_numpy(arr.transpose(2,0,1).copy())

def extract(path,m,samples=48):
    started=time.perf_counter();c=av.open(str(path));s=c.streams.video[0]
    duration=float(s.duration*s.time_base) if s.duration else c.duration/1e6
    origin=float((s.start_time or 0)*s.time_base)
    targets=np.linspace(0,max(0,duration-1/float(s.average_rate or 30)),samples)
    frames=[];actual=[];pos=0
    for f in c.decode(s):
        if f.pts is None:continue
        t=float(f.pts*s.time_base)-origin
        if pos<samples and t>=targets[pos]:
            x=tensor(f.to_image())
            while pos<samples and t>=targets[pos]:frames.append(x);actual.append(t);pos+=1
        if pos==samples:break
    c.close()
    if not frames:raise ValueError('No decodable frames')
    while len(frames)<samples:frames.append(frames[-1]);actual.append(actual[-1])
    emb=[]
    with torch.inference_mode():
        for i in range(0,samples,8):emb.append(m(torch.stack(frames[i:i+8])).numpy())
    z=np.concatenate(emb)
    features=np.concatenate([z.mean(0),z.std(0),z.max(0),np.abs(np.diff(z,axis=0)).mean(0)]).astype(np.float32)
    return features,dict(duration_sec=duration,sampled_timestamps_sec=actual,elapsed_sec=time.perf_counter()-started)

def score(x,weights):
    # Numeric-only checkpoint; no pickle needed for deployment.
    z=(x-weights['mean'])/weights['scale']
    logit=float((z@weights['coef'].T).item()+weights['intercept'][0])
    return float(1/(1+np.exp(-np.clip(logit,-50,50))))

def main():
    p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('--output');p.add_argument('--threads',type=int,default=4);a=p.parse_args()
    cfg=json.loads((ROOT/'config.json').read_text(encoding='utf8'))
    m=encoder(ROOT/'resnet18_imagenet.pth',a.threads)
    x,meta=extract(a.video,m,cfg['samples_per_video']);w=np.load(ROOT/'classifier.npz',allow_pickle=False);s=score(x,w)
    r=dict(model_version=cfg['model_version'],video=str(Path(a.video).resolve()),verdict='violation' if s>=cfg['decision_threshold'] else 'no_violation',violation_score=s,score_is_calibrated_probability=False,processing_sec=meta['elapsed_sec'],sampled_frames=cfg['samples_per_video'],event_localization_supported=False,limitation='Experimental whole-video classifier. No event class, timecode, rule, or independently validated 95% accuracy.')
    out=json.dumps(r,ensure_ascii=False,indent=2);print(out)
    if a.output:Path(a.output).write_text(out,encoding='utf8')
if __name__=='__main__':main()
