"""Inference for train-only selected Qwen/visual readout, with existing event head."""
from pathlib import Path
import sys,json,time,importlib.util
from observations import ROOT,WORK,LocalQwen4,vectors,digest,SPEC_HASH,np,sample
sys.path.insert(0,str(ROOT.parent/'forte_improved'))
import data
from search import scores
import joblib
sys.path.insert(0,str(ROOT.parent/'forte_transfer30'))
from event_head import EventHead

def display_name(selection):
    names=[]
    labels={'qwen2':['Qwen3-VL-2B'],'qwen4_summary':['Qwen3-VL-4B'],'qwen4_order':['Qwen3-VL-4B'],
            'dino_motion':['DINOv2'],'dino_r3d':['DINOv2','R3D18'],'r3d':['R3D18'],'resnet':['ResNet18']}
    for m in selection['candidate']['members']:
        for n in labels[m['feature']]:
            if n not in names:names.append(n)
    return ' + '.join(names)

class Predictor:
    def __init__(self,reuse_cache=True,model_dir=None):
        self.model_dir=Path(model_dir) if model_dir else ROOT
        lock=json.loads((self.model_dir/'selection_locked.json').read_text(encoding='utf8'))
        assert digest(self.model_dir/'classifier30.joblib')==lock['checkpoint_sha256']
        assert lock['feature_spec_hash']==SPEC_HASH
        self.package=joblib.load(self.model_dir/'classifier30.joblib');self.selection=self.package['selection']
        self.encoders={};self.name=display_name(self.selection);self.event_head=EventHead();self.reuse_cache=reuse_cache

    def predict(self,path,progress=lambda text:None):
        needed={m['feature'] for m in self.selection['candidate']['members']};features={};metas={};observations=None
        fingerprint=digest(path);cached={}
        if needed & {'qwen4_summary','qwen4_order'}:
            cache=ROOT/'feature_cache'/(fingerprint+'_'+SPEC_HASH[:12]+'.json');cached['qwen4']=self.reuse_cache and cache.exists()
            if cached['qwen4']:
                progress('Qwen: осы видеоның бұрын есептелген белгілерін оқу')
                record=json.loads(cache.read_text(encoding='utf8'));assert record['sha256']==fingerprint and record['spec_hash']==SPEC_HASH
                images,times,duration=sample(path);np.testing.assert_allclose(times,record['timestamps_sec'],atol=1e-8)
                record['evidence_views']=[dict(timestamp_sec=t,frames=[dict(actual_sec=t,image_data_url=url,view_kind=kind)
                    for url,kind in zip(pair,['full','desk_crop'])]) for t,pair in zip(times,images)]
                record['original_extraction_elapsed_sec']=record.pop('elapsed_sec');record['reused_cached_frozen_observations']=True
            else:
                progress('Qwen3-VL-4B: 6 сәттегі толық кадр мен үстел көрінісін талдау')
                with LocalQwen4(WORK/'qwen4_web.server.log') as q:record=q.extract(path,include_images=True)
            features.update(vectors(record));observations=record
            metas['qwen4']={k:record[k] for k in ['spec_hash','timestamps_sec','duration_sec']}
            metas['qwen4']['views_per_timestamp']=2
        if 'qwen2' in needed:
            progress('Qwen3-VL-2B: көрінетін әрекеттерді талдау')
            spec=importlib.util.spec_from_file_location('forte_qwen2_features',ROOT.parent/'forte_qwen_trained/features.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            with module.LocalQwen(WORK/'qwen2_web.server.log') as q:record=q.extract(path)
            features['qwen2']=module.vector(record['states']);observations=record
            metas['qwen2']={'sampled_timestamps_sec':record['sampled_timestamps_sec'],'spec_hash':record['spec_hash']}
        # DINO is also required by the separately trained event localization/type head.
        progress('DINOv2: оқиға түрі мен кадрлардың уақытын бағалау')
        cache=ROOT.parent/'forte_improved/feature_cache'/(fingerprint+'_dino24_dual_v1.npz');cached['dino']=self.reuse_cache and cache.exists()
        if cached['dino']:
            with np.load(cache,allow_pickle=False) as d:sequence=d['sequence'];patches=d['patches'];meta=json.loads(str(d['metadata']))
            assert meta['spec']==data.SPEC and sequence.shape==(24,2,768)
        else:
            if 'dino' not in self.encoders:self.encoders['dino']=data.encoder()
            sequence,meta,patches=data.extract(path,self.encoders['dino'])
        features.update(data.pooled(sequence,patches));metas['dino']=meta
        if needed & {'r3d','dino_r3d','resnet'}:
            sys.path.insert(0,str(ROOT.parent/'forte_model'))
            import temporal
            import model as old
            if needed & {'r3d','dino_r3d'}:
                progress('R3D18: видео қозғалысын талдау')
                cache=ROOT.parent/'forte_model/feature_cache'/(fingerprint+'_r3d18_8x16_fullresize_v1.npz');cached['r3d']=self.reuse_cache and cache.exists()
                if cached['r3d']:
                    with np.load(cache,allow_pickle=False) as d:features['r3d']=d['features'];metas['r3d']=json.loads(str(d['metadata']))
                    assert features['r3d'].shape==(1536,)
                else:
                    if 'r3d' not in self.encoders:self.encoders['r3d']=temporal.video_encoder()
                    features['r3d'],metas['r3d']=temporal.video_features(path,self.encoders['r3d'])
            if 'resnet' in needed:
                progress('ResNet18: кадрларды талдау')
                cache=ROOT.parent/'forte_model/feature_cache'/(fingerprint+'_resnet18_48_fullresize_v1.npz');cached['resnet']=self.reuse_cache and cache.exists()
                if cached['resnet']:
                    with np.load(cache,allow_pickle=False) as d:features['resnet']=d['features'];metas['resnet']=json.loads(str(d['metadata']))
                    assert features['resnet'].shape==(2048,)
                else:
                    if 'resnet' not in self.encoders:self.encoders['resnet']=old.encoder(old.ROOT/'resnet18_imagenet.pth')
                    features['resnet'],metas['resnet']=old.extract(path,self.encoders['resnet'],48)
        if 'dino_r3d' in needed:features['dino_r3d']=np.concatenate([features['dino_motion'],features['r3d']])
        c=self.selection['candidate'];ps=[float(scores(m,features[d['feature']][None,:])[0]) for d,m in zip(c['members'],self.package['models'])]
        video_score=float(np.average(ps,weights=c['weights']));threshold=self.selection['threshold']
        analysis=self.event_head.predict(sequence,meta['sampled_timestamps_sec'],meta['duration_sec'],video_score>=threshold)
        times=meta['sampled_timestamps_sec'];review=[[times[i]] for i in np.linspace(0,len(times)-1,8).astype(int)]
        return dict(score=video_score,threshold=threshold,model=self.name,model_version=self.package['version'],
            review_timestamps_sec=review,sampling={k:{a:b for a,b in v.items() if a!='elapsed_sec'} for k,v in metas.items()},
            member_scores=ps,member_weights=c['weights'],event_analysis=analysis,qwen_observations=observations,feature_cache_used=cached)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('--output');p.add_argument('--fresh',action='store_true');p.add_argument('--model-dir');a=p.parse_args()
    started=time.monotonic();result=Predictor(reuse_cache=not a.fresh,model_dir=a.model_dir).predict(Path(a.video),progress=lambda text:print(text,flush=True))
    result['processing_sec']=time.monotonic()-started
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if a.output:Path(a.output).write_text(text,encoding='utf8')
    else:print(text)
