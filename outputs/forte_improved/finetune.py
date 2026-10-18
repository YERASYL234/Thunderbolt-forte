"""Real partial CNN fine-tuning: ResNet18 layer4 and video MIL head, fixed recipe."""
from data import *
from torchvision.models import resnet18
from search import evaluate
import argparse,copy
from torch import nn
RECIPE=dict(version='resnet18_tail_mil_v1',samples=24,resize=160,views=['full','work_area'],
            train_frames_per_view=4,epochs=10,batch_videos=4,tail_learning_rate=.0001,
            head_learning_rate=.001,weight_decay=.01,seed=SEED,
            trained_layers='ResNet18 layer4 and 1024-to-1 mean+max video head',batchnorm='frozen running statistics',
            backbone_early_layers='frozen pretrained conv1 through layer3',threshold=.5,
            model_selection='Fixed recipe, no early stopping or tuning on validation folds')
def base():
    m=resnet18(weights=None)
    m.load_state_dict(torch.load(ROOT.parent/'forte_model/resnet18_imagenet.pth',map_location='cpu',weights_only=True))
    return m
def early(m):return nn.Sequential(m.conv1,m.bn1,m.relu,m.maxpool,m.layer1,m.layer2,m.layer3).eval()
def sample_maps(path,stem):
    frames=[];times=[]
    with av.open(str(path)) as c:
        s=c.streams.video[0];dur=float(s.duration*s.time_base) if s.duration else c.duration/1e6
        origin=float((s.start_time or 0)*s.time_base);targets=np.linspace(.01*dur,.99*dur,24);pos=0
        for f in c.decode(s):
            if f.pts is None:continue
            t=float(f.pts*s.time_base)-origin
            if pos<24 and t>=targets[pos]:
                im=f.to_image().convert('RGB');w,h=im.size;views=[]
                for box in SPEC['views'].values():
                    crop=im.crop((round(w*box[0]),round(h*box[1]),round(w*box[2]),round(h*box[3]))).resize((160,160),Image.Resampling.BILINEAR)
                    a=np.asarray(crop,dtype=np.float32)/255
                    a=(a-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
                    views.append(torch.from_numpy(a.transpose(2,0,1).copy()))
                x=torch.stack(views)
                while pos<24 and t>=targets[pos]:frames.append(x);times.append(t);pos+=1
            if pos==24:break
    if len(frames)!=24:raise ValueError('Incomplete video sampling')
    inp=torch.stack(frames).reshape(-1,3,160,160);result=[]
    with torch.inference_mode():
        for i in range(0,len(inp),8):result.append(stem(inp[i:i+8]).cpu())
    return torch.cat(result).reshape(24,2,256,10,10).half(),times
class TailModel(nn.Module):
    def __init__(self,source):
        super().__init__();self.tail=copy.deepcopy(source.layer4);self.head=nn.Linear(1024,1)
        self.norm=nn.LayerNorm(1024) if RECIPE.get('pooled_layernorm',False) else nn.Identity()
        nn.init.normal_(self.head.weight,std=.01);nn.init.zeros_(self.head.bias)
    def train(self,mode=True):
        super().train(mode)
        for module in self.tail.modules():
            if isinstance(module,nn.BatchNorm2d):module.eval()
        return self
    def forward(self,x):
        b,t=x.shape[:2];z=self.tail(x.reshape(-1,256,10,10));z=z.mean((2,3)).reshape(b,t,512)
        return self.head(self.norm(torch.cat([z.mean(1),z.max(1).values],1))).squeeze(-1)
def infer(m,maps,indices):
    m.eval();p=[]
    with torch.inference_mode():
        for i in indices:
            if RECIPE.get('eval_stratified_bags',False):
                # Match the 4-timepoint/2-view training bag; average all six temporal offsets.
                bags=[maps[i][[j,j+6,j+12,j+18]].reshape(8,256,10,10) for j in range(6)]
                p.append(float(torch.sigmoid(m(torch.stack(bags).float())).mean().item()))
            else:
                x=maps[i].float().reshape(1,48,256,10,10)
                p.append(float(torch.sigmoid(m(x)).item()))
    return np.asarray(p)
def train_one(maps,y,indices,source,seed):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);m=TailModel(source).train()
    opt=torch.optim.AdamW([{'params':m.tail.parameters(),'lr':RECIPE['tail_learning_rate']},
                           {'params':list(m.head.parameters())+list(m.norm.parameters()),'lr':RECIPE['head_learning_rate']}],weight_decay=RECIPE['weight_decay'])
    counts=np.bincount(y[indices],minlength=2);loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(counts[0]/counts[1])))
    losses=[]
    for epoch in range(RECIPE['epochs']):
        m.train();order=rng.permutation(indices);total=0
        for start in range(0,len(order),RECIPE['batch_videos']):
            ids=order[start:start+RECIPE['batch_videos']];batch=[]
            for i in ids:
                # One random frame from each temporal quarter, both views; no event-label sampling.
                take=np.array([rng.integers(j*6,(j+1)*6) for j in range(4)])
                x=maps[i][take].reshape(8,256,10,10).float()
                if RECIPE.get('flip_augmentation',True) and rng.random()<.5:x=x.flip(-1)
                batch.append(x)
            target=torch.tensor(y[ids],dtype=torch.float32);opt.zero_grad(set_to_none=True)
            loss=loss_fn(m(torch.stack(batch)),target);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.)
            opt.step();total+=float(loss.detach())*len(ids)
        losses.append(total/len(order))
        if epoch in (0,4,9,14,19):print('EPOCH',epoch+1,'loss',round(losses[-1],4),flush=True)
    changed=float((m.tail[0].conv1.weight.detach()-source.layer4[0].conv1.weight.detach()).abs().max())
    assert changed>0,'Backbone tail did not update'
    return m,dict(epoch_losses=losses,max_abs_layer4_conv1_weight_change=changed)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stable',action='store_true');ap.add_argument('--matched',action='store_true');args=ap.parse_args()
    prefix='finetune_matched' if args.matched else ('finetune_stable' if args.stable else 'finetune')
    if args.stable or args.matched:
        RECIPE.update(version='resnet18_tail_mil_stable_v2',epochs=20,tail_learning_rate=.00003,
                      head_learning_rate=.0003,pooled_layernorm=True,flip_augmentation=False,
                      experiment_note='Added after observing unstable training loss in v1; developmental experiment, no independent test')
    if args.matched:RECIPE.update(version='resnet18_tail_mil_matched_v3',eval_stratified_bags=True,
        experiment_note='Match inference bag size to training: six deterministic stratified bags averaged; added after reviewing v2 train/inference pooling mismatch')
    vs=videos();y=np.array([int(v['source_verdict']=='violation') for v in vs]);folds=np.array([v['provisional_fold'] for v in vs])
    cache=ROOT/'tail_feature_cache';cache.mkdir(exist_ok=True);source=base();stem=early(source);maps=[];timings=[]
    (ROOT/(prefix+'_recipe.json')).write_text(json.dumps(RECIPE,indent=2),encoding='utf8')
    for i,v in enumerate(vs):
        dest=cache/(v['sha256']+'_stem160_24x2.pt');started=time.monotonic()
        if dest.exists():d=torch.load(dest,map_location='cpu',weights_only=True)
        else:
            assert sha(DS/v['path'])==v['sha256'];x,t=sample_maps(DS/v['path'],stem)
            d={'maps':x,'times':t,'sha256':v['sha256']};torch.save(d,dest)
        assert d['sha256']==v['sha256'];maps.append(d['maps']);timings.append(time.monotonic()-started)
        print('CNN_MAPS',i+1,'/40',round(timings[-1],1),'seconds',flush=True)
    p=np.zeros(len(y));details=[]
    for f in sorted(set(folds)):
        tr=np.flatnonzero(folds!=f);te=np.flatnonzero(folds==f);print('FINETUNE_FOLD',int(f),flush=True)
        m,audit=train_one(maps,y,tr,source,SEED+int(f));p[te]=infer(m,maps,te)
        if args.matched:
            saved=WORK/'finetune_validation';saved.mkdir(exist_ok=True)
            torch.save({'state_dict':m.state_dict(),'recipe':RECIPE,'train_indices':tr.tolist()},saved/(prefix+f'_fold{int(f)}.pt'))
        details.append(dict(fold=int(f),metrics=evaluate(y[te],p[te]),training_audit=audit,training_indices=tr.tolist(),validation_indices=te.tolist()))
        print('FOLD_RESULT',int(f),json.dumps(details[-1]['metrics']),flush=True)
    print('FINAL_FIT',flush=True);final,audit=train_one(maps,y,np.arange(len(y)),source,SEED+100)
    checkpoint=ROOT/('finetuned_matched_tail.pt' if args.matched else ('finetuned_stable_tail.pt' if args.stable else 'finetuned_tail.pt'));torch.save({'state_dict':final.state_dict(),'recipe':RECIPE,'training_hashes':[v['sha256'] for v in vs]},checkpoint)
    restored=TailModel(source);restored.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict'])
    np.testing.assert_allclose(infer(final,maps,[0,1]),infer(restored,maps,[0,1]),atol=1e-7)
    report=dict(metrics=evaluate(y,p),evaluation='Fixed-recipe 5-fold DEVELOPMENT evaluation; no independent final test',recipe=RECIPE,
                fold_details=details,final_training_audit=audit,limitations=['40 videos, similar scenes/actors across folds.','Source labels not independently adjudicated.',
                'Only layer4 and video head fine-tuned, not the whole network.','No event class or exact localization.','Experimental, not operationally validated.'])
    rows=[dict(video_id=v['video_id'],sha256=v['sha256'],fold=int(folds[i]),true_label=int(y[i]),score=float(p[i]),predicted_label=int(p[i]>=.5),correct=bool((p[i]>=.5)==y[i])) for i,v in enumerate(vs)]
    (ROOT/(prefix+'_metrics.json')).write_text(json.dumps(report,indent=2),encoding='utf8')
    (ROOT/(prefix+'_oof_predictions.json')).write_text(json.dumps(rows,indent=2),encoding='utf8')
    print('FINETUNE_RESULT',json.dumps(report['metrics']),flush=True)
if __name__=='__main__':main()
