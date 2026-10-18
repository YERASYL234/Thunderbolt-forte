from common import *
import math
from torch import nn
import torch.nn.functional as F

class TemporalMIL(nn.Module):
    """Each instance is 8 ordered dual-view frames (4 s at 2 FPS)."""
    def __init__(self,config=None):
        super().__init__();self.config=dict(CONFIG if config is None else config)
        a,b=self.config['hidden_channels'];p=self.config['dropout']
        self.temporal=nn.Sequential(nn.Conv1d(768,a,3),nn.GELU(),nn.Dropout(p),
                                    nn.Conv1d(a,b,3),nn.GELU(),nn.Dropout(p))
        self.head=nn.Linear(b,1)
    def forward(self,windows):return self.head(self.temporal(windows).mean(-1)).squeeze(-1)
    def pool(self,logits):
        k=max(1,math.ceil(len(logits)*self.config['mil_top_fraction']))
        return logits.topk(k).values.mean()

def make_windows(z,times,duration,config=None):
    cfg=CONFIG if config is None else config
    x=torch.as_tensor(z.reshape(len(z),768),dtype=torch.float32)
    n=cfg['window_frames'];stride=cfg['stride_frames']
    if len(x)<n:x=torch.cat([x,x[-1:].repeat(n-len(x),1)])
    starts=list(range(0,len(x)-n+1,stride))
    if starts[-1]!=len(x)-n:starts.append(len(x)-n)
    windows=torch.stack([x[s:s+n].T for s in starts])
    bounds=np.array([[float(times[min(s,len(times)-1)]),min(duration,float(times[min(s+n-1,len(times)-1)])+.5)] for s in starts])
    return windows,bounds

def point_mask(bounds,points,radius):
    centers=bounds.mean(1);mask=np.zeros(len(bounds),bool)
    for t in points:
        selected=np.abs(centers-t)<=radius
        if not selected.any():selected[np.argmin(np.abs(centers-t))]=True
        mask|=selected
    return torch.from_numpy(mask)

def loss_for_bag(model,logits,label,point_targets,class_weights):
    video=F.binary_cross_entropy_with_logits(model.pool(logits),logits.new_tensor(float(label)))*class_weights[label]
    point=F.softplus(-logits[point_targets]).mean() if label and point_targets.any() else logits.sum()*0
    p=logits.sigmoid();smooth=((p[1:]-p[:-1])**2).mean() if len(p)>1 else logits.sum()*0
    loss=video+model.config['lambda_point']*point+model.config['mu_smooth']*smooth
    return loss,torch.stack([video.detach(),point.detach(),smooth.detach()])

def normalizer(sequences,indices):
    # Equal weight per training video; no held-out feature statistics.
    means=torch.stack([sequences[i].mean(0) for i in indices])
    squares=torch.stack([(sequences[i]**2).mean(0) for i in indices])
    mean=means.mean(0);std=(squares.mean(0)-mean**2).clamp_min(.01).sqrt()
    return mean,std

def event_candidates(bounds,scores,threshold=.5):
    intervals=[]
    for (start,end),score in zip(bounds,scores):
        if score<threshold:continue
        if intervals and start<=intervals[-1]['end_sec']:
            intervals[-1]['end_sec']=float(end);intervals[-1]['score']=max(intervals[-1]['score'],float(score))
        else:intervals.append(dict(start_sec=float(start),end_sec=float(end),score=float(score),
                                   event_type='suspicious_action_untyped',display_type='Типі анықталмаған күмәнді әрекет',
                                   boundaries='candidate_4s_windows_not_verified_event_intervals'))
    return intervals
