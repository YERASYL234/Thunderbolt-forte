"""Local inference for the locked train30 selection; no annotation lookup."""
from pathlib import Path
import sys, json, argparse, time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'forte_improved'))
import data
from data import np, torch
from search import scores
import finetune as ft
import joblib
from event_head import EventHead

def display_name(selection):
    labels={'r3d':['R3D18'],'resnet':['ResNet18'],'resnet_tail':['ResNet18 fine-tuned'],
            'dino_dual':['DINOv2'],'dino_motion':['DINOv2'],'dino_r3d':['DINOv2','R3D18']}
    names=[]
    for r in selection['members']:
        for name in labels[r['candidate']['feature']]:
            if name not in names:names.append(name)
    return ' + '.join(names)

class Predictor:
    def __init__(self):
        lock=json.loads((ROOT/'selection_locked.json').read_text(encoding='utf8'))
        assert data.sha(ROOT/'classifier30.joblib')==lock['checkpoint_sha256']
        assert data.sha(ROOT/'finetuned_tail30.pt')==lock['tail_checkpoint_sha256']
        self.package=joblib.load(ROOT/'classifier30.joblib')
        self.selection=self.package['selection'];self.encoders={}
        self.name=display_name(self.selection)
        self.event_head=EventHead()

    def predict(self,path,progress=lambda text:None):
        needed={r['candidate']['feature'] for r in self.selection['members']}
        features={};metas={};tail_score=None
        if self.event_head is not None or any(k.startswith('dino') for k in needed):
            progress('DINOv2: толық кадр мен үстел аймағын талдау')
            if 'dino' not in self.encoders:self.encoders['dino']=data.encoder()
            z,meta,patches=data.extract(path,self.encoders['dino'])
            features.update(data.pooled(z,patches));metas['dino']=meta
            event_sequence=z;event_meta=meta
        if needed & {'r3d','dino_r3d','resnet'}:
            sys.path.insert(0,str(ROOT.parent/'forte_model'))
            import temporal
            import model as old
            if needed & {'r3d','dino_r3d'}:
                progress('R3D18: видео үзінділерін талдау')
                if 'r3d' not in self.encoders:self.encoders['r3d']=temporal.video_encoder()
                features['r3d'],metas['r3d']=temporal.video_features(path,self.encoders['r3d'])
            if 'resnet' in needed:
                progress('ResNet18: видео кадрларын талдау')
                if 'resnet' not in self.encoders:self.encoders['resnet']=old.encoder(old.ROOT/'resnet18_imagenet.pth')
                features['resnet'],metas['resnet']=old.extract(path,self.encoders['resnet'],48)
        if 'dino_r3d' in needed:features['dino_r3d']=np.concatenate([features['dino_motion'],features['r3d']])
        if 'resnet_tail' in needed:
            progress('Қайта оқытылған ResNet18: кадрларды талдау')
            if 'tail' not in self.encoders:
                ckpt=torch.load(ROOT/'finetuned_tail30.pt',map_location='cpu',weights_only=True)
                ft.RECIPE=ckpt['recipe'];base=ft.base();net=ft.TailModel(base)
                net.load_state_dict(ckpt['state_dict']);net.eval()
                self.encoders['stem']=ft.early(base);self.encoders['tail']=net
            maps,times=ft.sample_maps(path,self.encoders['stem'])
            tail_score=float(ft.infer(self.encoders['tail'],[maps],[0])[0])
            metas['tail']={'sampled_timestamps_sec':times}
        ps=[]
        for row,m in zip(self.selection['members'],self.package['models']):
            c=row['candidate']
            ps.append(tail_score if c['kind']=='finetune' else float(scores(m,features[c['feature']][None,:])[0]))
        if 'r3d' in metas:
            review=metas['r3d']['clip_timestamps_sec']
        else:
            times=next(iter(metas.values()))['sampled_timestamps_sec']
            review=[[times[i]] for i in np.linspace(0,len(times)-1,8).astype(int)]
        video_score=float(np.mean(ps))
        progress('Оқиға түрлері мен күмәнді үзінділерді бағалау')
        event_analysis=self.event_head.predict(event_sequence,event_meta['sampled_timestamps_sec'],event_meta['duration_sec'],video_score>=self.selection['threshold'])
        return dict(score=video_score,threshold=self.selection['threshold'],model=self.name,
                    model_version=self.package['model_version'],review_timestamps_sec=review,
                    sampling={k:{a:b for a,b in meta.items() if a not in ['elapsed_sec']} for k,meta in metas.items()},
                    member_scores=ps,event_analysis=event_analysis)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('video');ap.add_argument('--output');a=ap.parse_args()
    start=time.monotonic();result=Predictor().predict(Path(a.video));result['processing_sec']=time.monotonic()-start
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if a.output:Path(a.output).write_text(text,encoding='utf8')
    print(text)
